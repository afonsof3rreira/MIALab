import os
import numpy as np


# featureList = []
# first_order_subL = []
# first_order_subL.append('la')
# first_order_subL.append('lolala')
# others = []
# others.append('lalal')
# featureList.append(first_order_subL)
# featureList.append(others)
#
# three = []
# three.append('la')
# three.append('lo')
# three.append('ka')
# three.append('kAo')
#
# featureList = {1: 'first_order_subL', 2: 'others'}
# featureList.update({5: three})
# # print(featureList)
#
# path = 'C:/Users/afons/PycharmProjects/MIAlab project/bin'
# filename = 'result_report'

def feature_writer(path: str, feature_dic: dict, running_time: float, filename: str):
    non_feature_list = ['skullstrip_pre', 'normalization_pre', 'registration_pre']

    feature_list = ['coordinates_feature', 'intensity_feature', 'gradient_intensity_feature',
                    'first_order_feature_parameters',
                    'HOG_feature', 'GLCM_features_parameters']

    if not os.path.exists(os.path.join(path, filename + '.txt')):

        with open(os.path.join(path, filename + '.txt'), 'w') as outfile:

            outfile.write('=' * 10 + feature_dic.get('experiment_name') + '=' * 10 + '\n\n')
            outfile.write('-' * 10 + ' Features tested ' + '-' * 10)
            outfile.write('\n' + '-' * 37 + '\n')

            # getting and writing all used features
            for key, values in feature_dic.items():

                if isinstance(values, dict) and key in feature_list:
                    outfile.write('\n' + '# ' + key)
                    for features, boolean in values.items():
                        if boolean:
                            outfile.write('\n' + '      # ' + features)

                    outfile.write(str() + '\n')

                elif key in feature_list:
                    outfile.write('\n' + '# ' + key)

            # getting and writing RF parameters
            n_estimators = feature_dic.get('n_estimators')
            max_depth = feature_dic.get('max_depth')
            outfile.write(
                '\n' + '-' * 10 + ' RF parameters ' + '-' * 11 + '\n' + '# n_estimators = ' + str(n_estimators)
                + '\n' + '# max_depth = ' + str(max_depth))

            # getting and writing non-feature parameters
            outfile.write('\n\n' + '-' * 10 + ' Non-feature parameters ' + '--')
            for key, values in feature_dic.items():
                if key in non_feature_list and values:
                    outfile.write('\n' + '# ' + key)

            # writing the computational time
            outfile.write('\n\n' + '-> Pipeline running time = ' + f'{round(running_time, 3)} second(s)')
