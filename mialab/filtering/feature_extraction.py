"""The feature extraction module contains classes for feature extraction."""
import sys

import SimpleITK as sitk
import numpy as np
import pymia.filtering.filter as fltr
import multiprocessing
import time


# from radiomics import shape.RadiomicsShape


class AtlasCoordinates(fltr.Filter):
    """Represents an atlas coordinates feature extractor."""

    def __init__(self):
        """Initializes a new instance of the AtlasCoordinates class."""
        super().__init__()

    def execute(self, image: sitk.Image, params: fltr.FilterParams = None) -> sitk.Image:
        """Executes a atlas coordinates feature extractor on an image.

        Args:
            image (sitk.Image): The image.
            params (fltr.FilterParams): The parameters (unused).

        Returns:
            sitk.Image: The atlas coordinates image
            (a vector image with 3 components, which represent the physical x, y, z coordinates in mm).

        Raises:
            ValueError: If image is not 3-D.
        """

        if image.GetDimension() != 3:
            raise ValueError('image needs to be 3-D')

        x, y, z = image.GetSize()

        # create matrix with homogenous indices in axis 3
        coords = np.zeros((x, y, z, 4))
        coords[..., 0] = np.arange(x)[:, np.newaxis, np.newaxis]
        coords[..., 1] = np.arange(y)[np.newaxis, :, np.newaxis]
        coords[..., 2] = np.arange(z)[np.newaxis, np.newaxis, :]
        coords[..., 3] = 1

        # reshape such that each voxel is one row
        lin_coords = np.reshape(coords, [coords.shape[0] * coords.shape[1] * coords.shape[2], 4])

        # generate transformation matrix
        tmpmat = image.GetDirection() + image.GetOrigin()
        tfm = np.reshape(tmpmat, [3, 4], order='F')
        tfm = np.vstack((tfm, [0, 0, 0, 1]))

        atlas_coords = (tfm @ np.transpose(lin_coords))[0:3, :]
        atlas_coords = np.reshape(np.transpose(atlas_coords), [z, y, x, 3], 'F')

        img_out = sitk.GetImageFromArray(atlas_coords)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'AtlasCoordinates:\n' \
            .format(self=self)


def percentile_subset(array, min, max):
    """ Outputs a subset of image array with gray levels in between, or equal to the "min"-th and "max"-th percentile

    Args:
        array (np.array): sorted 3D array
        min (float): lower quantile value
        max (float): upper quantile value

    Returns:
        np.array: A vector containing values between min and max, inclusive
    """

    list = []

    for i in range(0, len(array)):

        if array.ndim != 1:

            for j in range(0, len(array)):
                for k in range(0, len(array)):

                    if min <= array[i][j][k] <= max:
                        list.append(array[i][j][k])

        elif min <= array[i] <= max:

            list.append(array[i])

    return np.asarray(list)


class selected_features:

    def __init__(self):
        self.featureList = []

        # TODO: comment features that are not used in first_order_texture_features_function
        self.featureList.append("mean")
        self.featureList.append("variance")
        self.featureList.append("sigma")
        self.featureList.append("skewness")
        self.featureList.append("kurtosis")
        #self.featureList.append("entropy")
        self.featureList.append("snr")
        self.featureList.append("min")
        self.featureList.append("max")
        self.featureList.append("range")
        self.featureList.append("percentile10th")
        self.featureList.append("percentile25th")
        self.featureList.append("percentile50th")
        self.featureList.append("percentile75th")
        self.featureList.append("percentile90th")
        # '-----added----'
        #self.featureList.append("inter-quartile range = p75 - p25")
        self.featureList.append("mean absolute deviation")
        self.featureList.append("robust mean absolute deviation")

    def getSelectedFeatures(self):
        return self.featureList


def first_order_texture_features_function(values):
    """Calculates first-order texture features.

    Args:
        values (np.array): The values to calculate the first-order texture features from.

    Returns:
        np.array: A vector containing the first-order texture features:

            - mean
            - variance
            - sigma
            - skewness
            - kurtosis
            - entropy
            - energy
            - snr
            - min
            - max
            - range
            - percentile10th
            - percentile25th
            - percentile50th = median
            - percentile75th
            - percentile90th
              -----added----
            - inter-quartile range = p75 - p25
            - mean absolute deviation
            - robust mean absolute deviation
    """
    eps = sys.float_info.epsilon  # to avoid division by zero

    mean = np.mean(values)
    std = np.std(values)
    snr = mean / std if std != 0 else 0
    min_ = np.min(values)
    max_ = np.max(values)
    numvalues = len(values)
    p = values / (np.sum(values) + eps)

    #   array containing indexes of np.values between 10-th and 90-th percentile

    values_p1090 = percentile_subset(np.sort(values), np.percentile(values, 10),
                                     np.percentile(values, 90))
    mean_p1090 = np.mean(values_p1090)
    numvalues_p1090 = len(values_p1090)

    return np.array([mean,
                     np.var(values),  # variance
                     std,
                     np.sqrt(numvalues * (numvalues - 1)) / (numvalues - 2) * np.sum((values - mean) ** 3) / (numvalues * std ** 3 + eps),  # adjusted Fisher-Pearson coefficient of skewness
                     np.sum((values - mean) ** 4) / (numvalues * std ** 4 + eps),  # kurtosis
                     #np.sum(-p * np.log2(p)),  # entropy
                     np.sum(p ** 2),  # energy (intensity histogram uniformity)
                     snr,
                     min_,
                     max_,
                     max_ - min_,
                     np.percentile(values, 10),
                     np.percentile(values, 25),
                     np.percentile(values, 50),
                     np.percentile(values, 75),
                     np.percentile(values, 90),
                     # -----added----
                     np.percentile(values, 75) - np.percentile(values, 25),
                     np.sum(values - mean) / numvalues,
                     #np.sum(values_p1090 - mean_p1090) / numvalues_p1090
                     ])


def firstOFeature_slice(padded_img, out_img, zz, z_offset, y_offset, x_offset):
    """Calculates 1st-order features for the zz-th 2-D "slice" of the whole brain image

            Args:
                padded_img (np.array): padded image from which features are extracted
                out_img (np.array): original image with shape (z, y, x, NP)
                    where NP = number of components per pixel
                zz (int): zz fixed axis value for which 2-D slice is obtained
                z_offset (int): z-axis offset
                y_offset (int): y-axis offset
                x_offset(int) : x-axis offset

            Returns:
                List[np.array, int]: A List element containing the zz-th feature-extracted-slice and the zz value
            """

    # np.shape(out_img) = (z, y, x, NP)
    # sub_dim: A tuple with dimensions (y, x, NP) where NP = number of components per pixel
    sub_dim = np.shape(out_img)[1:]

    mat = np.zeros(sub_dim)

    for yy in range(sub_dim[0]):
        for xx in range(sub_dim[1]):
            val = first_order_texture_features_function(
                padded_img[zz:zz + z_offset, yy:yy + y_offset, xx:xx + x_offset])
            mat[yy][xx] = val

    return [mat, zz]


def firstOFeature_slice_aux(args):
    """auxiliary function that enables using func_2 with a single argument (List)"""
    return firstOFeature_slice(*args)


class NeighborhoodFeatureExtractor(fltr.Filter):
    """Represents a feature extractor filter, which works on a neighborhood."""

    def __init__(self, kernel=(3, 3, 3), function_=first_order_texture_features_function):
        """Initializes a new instance of the NeighborhoodFeatureExtractor class."""
        super().__init__()
        self.neighborhood_radius = 3
        self.kernel = kernel
        self.function = function_

    def execute(self, image: sitk.Image, params: fltr.FilterParams = None,
                multiprocessing_features: bool = False) -> sitk.Image:
        """Executes a neighborhood feature extractor on an image.

        Args:
            image (sitk.Image): The image.
            params (fltr.FilterParams): The parameters (unused).
            multiprocessing_features: uses multiprocessing for feature extraction if specified as True

        Returns:
            sitk.Image: The normalized image.

        Raises:
            ValueError: If image is not 3-D.
        """
        # image.GetSize() = (197, 233, 189)
        if image.GetDimension() != 3:
            raise ValueError('image needs to be 3-D')

        # test the function and get the output dimension for later reshaping
        function_output = self.function(np.array([1, 2, 3]))
        if np.isscalar(function_output):  # how can this ever be a scalar if first_order_features output nd.arrays?
            img_out = sitk.Image(image.GetSize(), sitk.sitkFloat32)  # image is 3D (N x M x C)
        elif not isinstance(function_output, np.ndarray):  # if function_output isn't nd.array (scalar nor array)
            raise ValueError('function must return a scalar or a 1-D np.ndarray')
        elif function_output.ndim > 1:  # if nd.array is non 1-D
            raise ValueError('function must return a scalar or a 1-D np.ndarray')
        elif function_output.shape[0] <= 1:  # if nd.array doesn't have at least 2 elements
            raise ValueError('function must return a scalar or a 1-D np.ndarray with at least two elements')
        else:  # ---------------(197, 233, 189)----------------------------number of features: int
            img_out = sitk.Image(image.GetSize(), sitk.sitkVectorFloat32, function_output.shape[0])
        # this last else will create an img_out which has a number of components per pixel = number of features
        #   i.e. another dimension will be added for the 3-D matrix, making it "4-D"
        # prove: "number of components per pixel = img_out.GetNumberOfComponentsPerPixel()
        #   img_out still has size = (197, 233, 189)
        # ------------------------------------------------- z    y    x
        img_out_arr = sitk.GetArrayFromImage(img_out)  # (189, 233, 197, 2)
        img_arr = sitk.GetArrayFromImage(image)  # (189, 233, 197)   shape is "swapped"
        z, y, x = img_arr.shape
        z_offset = self.kernel[2]
        y_offset = self.kernel[1]
        x_offset = self.kernel[0]
        pad = ((0, z_offset), (0, y_offset), (0, x_offset))
        #   img_arr is extended by adding 3 sheets, 3 rows and 3 columns
        img_arr_padded = np.pad(img_arr, pad, 'symmetric')  # (192, 236, 200)

        start = time.perf_counter()

        # with concurrent.futures.ProcessPoolExecutor() as executor:
        # params_list = list_maker(img_arr_padded, y, y_offset, z, z_offset, x, x_offset)
        if not multiprocessing_features:

            for xx in range(x):
                for yy in range(y):
                    for zz in range(z):
                        print('x, y, z = ' + str(xx) + ' ' + str(yy) + ' ' + str(zz))
                        val = self.function(img_arr_padded[zz:zz + z_offset, yy:yy + y_offset, xx:xx + x_offset])
                        img_out_arr[zz, yy, xx] = val

        else:

            rets = []
            # builds sets of arguments to be fed to pools
            for zz in range(z):
                rets.append([img_arr_padded, img_out_arr, zz, z_offset, y_offset, x_offset])

            # TODO: change argument to (multiprocessing.cpu_count() - 1) when using own computer
            # as it sets a limit of cpu cores to be used, without overloading hardware
            p = multiprocessing.Pool(multiprocessing.cpu_count())

            result = p.map(firstOFeature_slice_aux, rets)

            p.close()
            p.join()

            # Assigns each obtained "zz-slice" to each row of the 4-D array output image
            # We have to check the whole result and assign slices in a sorted way since pools are not synchronous
            for zz in range(z):
                for zzi in range(z):

                    if result[zzi][1] == zz:
                        img_out_arr[zz, :, :] = result[zzi][0]
                        break

        finish = time.perf_counter()
        print(f'Finished in {round(finish - start, 2)} seconds(s)')

        img_out = sitk.GetImageFromArray(img_out_arr)
        img_out.CopyInformation(image)

        return img_out

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'NeighborhoodFeatureExtractor:\n' \
            .format(self=self)


class RandomizedTrainingMaskGenerator:
    """Represents a training mask generator.

    A training mask is an image with intensity values 0 and 1, where 1 represents masked.
    Such a mask can be used to sample voxels for training.
    """

    @staticmethod
    def get_mask(ground_truth: sitk.Image,
                 ground_truth_labels: list,
                 label_percentages: list,
                 background_mask: sitk.Image = None) -> sitk.Image:
        """Gets a training mask.

        Args:
            ground_truth (sitk.Image): The ground truth image.
            ground_truth_labels (list of int): The ground truth labels,
                where 0=background, 1=label1, 2=label2, ..., e.g. [0, 1]
            label_percentages (list of float): The percentage of voxels of a corresponding label to extract as mask,
                e.g. [0.2, 0.2].
            background_mask (sitk.Image): A mask, where intensity 0 indicates voxels to exclude independent of the label.

        Returns:
            sitk.Image: The training mask.
        """

        # initialize mask
        ground_truth_array = sitk.GetArrayFromImage(ground_truth)
        mask_array = np.zeros(ground_truth_array.shape, dtype=np.uint8)

        # exclude background
        if background_mask is not None:
            background_mask_array = sitk.GetArrayFromImage(background_mask)
            background_mask_array = np.logical_not(background_mask_array)
            ground_truth_array = ground_truth_array.astype(float)  # convert to float because of np.nan
            ground_truth_array[background_mask_array] = np.nan

        for label_idx, label in enumerate(ground_truth_labels):
            indices = np.transpose(np.where(ground_truth_array == label))
            np.random.shuffle(indices)

            no_mask_items = int(indices.shape[0] * label_percentages[label_idx])

            for no in range(no_mask_items):
                x = indices[no][0]
                y = indices[no][1]
                z = indices[no][2]
                mask_array[x, y, z] = 1  # this is a masked item

        mask = sitk.GetImageFromArray(mask_array)
        mask.SetOrigin(ground_truth.GetOrigin())
        mask.SetDirection(ground_truth.GetDirection())
        mask.SetSpacing(ground_truth.GetSpacing())

        return mask
