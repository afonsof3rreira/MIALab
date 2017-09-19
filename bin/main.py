"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
"""
import argparse
import datetime
import os
import sys
import timeit

import numpy as np
from tensorflow.python.platform import app
import SimpleITK as sitk

import mialab.classifier.decision_forest as df
import mialab.data.conversion as conversion
import mialab.data.structure as structure
import mialab.utilities as util

FLAGS = None  # the program flags
IMAGE_KEYS = [structure.BrainImageTypes.T1, structure.BrainImageTypes.T2, structure.BrainImageTypes.GroundTruth]  # the list of images we will load


def main(_):
    """Brain tissue segmentation using decision forests.

    The main routine executes the medical image analysis pipeline:
        - Image loading
        - ...
    """

    # load atlas images
    util.load_atlas_images(FLAGS.data_atlas_dir)

    print('-' * 5, 'Training...')
    # load images for training
    images = util.process_batch(FLAGS.data_train_dir, IMAGE_KEYS, True)

    # generate feature matrix and label vector
    data_train = np.concatenate([img.feature_matrix[0] for img in images])
    labels_train = np.concatenate([img.feature_matrix[1] for img in images])

    # initialize decision forest parameters
    params = df.DecisionForestParameters()
    params.num_classes = 4
    params.num_features = images[0].feature_matrix[0].shape[1]
    params.num_trees = 20
    params.max_nodes = 1000

    # generate a model directory (use datetime to ensure that the directory is empty)
    # we need an empty directory because TensorFlow will continue training an existing model if it is not empty
    t = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
    model_dir = os.path.join(FLAGS.model_dir, t)
    os.makedirs(model_dir, exist_ok=True)
    params.model_dir = model_dir
    print(params)

    forest = df.DecisionForest(params)
    start_time = timeit.default_timer()
    forest.train(data_train, labels_train)
    print(' Time elapsed:', timeit.default_timer() - start_time, 's')

    print('-' * 5, 'Testing...')
    result_dir = os.path.join(FLAGS.result_dir, t)
    os.makedirs(result_dir, exist_ok=True)

    # initialize evaluator
    evaluator = util.init_evaluator(result_dir)

    # load images for testing
    images_test = util.process_batch(FLAGS.data_test_dir, IMAGE_KEYS, False)

    for img in images_test:
        data_test = img.feature_matrix[0]
        labels_test = img.feature_matrix[1]

        print('-' * 10, 'Testing', img.id_)
        start_time = timeit.default_timer()
        probabilities, predictions = forest.predict(data_test)
        print(' Time elapsed:', timeit.default_timer() - start_time, 's')

        # print feature importances if calculated
        if params.report_feature_importances:
            results = forest.evaluate(data_test, labels_test)
            for key in sorted(results):
                print('%s: %s' % (key, results[key]))

        # convert prediction and probabilities back to SimpleITK images
        image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.uint8),
                                                                        img.image_properties)

        image_probabilities = conversion.NumpySimpleITKImageBridge.convert_to_vector_image(probabilities,
                                                                                           img.image_properties)

        # evaluate segmentation without post-processing
        evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

        # post-process segmentation and evaluate with post-processing
        image_post_processed = util.post_process(img, image_prediction, image_probabilities)
        evaluator.evaluate(image_post_processed, img.images[structure.BrainImageTypes.GroundTruth], img.id_ + '-DCRF')

        # save results
        sitk.WriteImage(image_prediction, os.path.join(result_dir, img.id_ + '_SEG.mha'), True)
        sitk.WriteImage(image_post_processed, os.path.join(result_dir, img.id_ + '_SEG-PP.mha'), True)


if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')
    parser.add_argument(
        '--model_dir',
        type=str,
        default=os.path.join(script_dir, 'mia-model'),
        help='Base directory for output models.'
    )

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.join(script_dir, 'mia-result'),
        help='Directory for results.'
    )

    parser.add_argument(
        '--data_atlas_dir',
        type=str,
        default=os.path.join(script_dir, '../data/atlas'),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_train_dir',
        type=str,
        default=os.path.join(script_dir, '../data/train/'),
        help='Directory with training data.'
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.join(script_dir, '../data/test/'),
        help='Directory with testing data.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)
