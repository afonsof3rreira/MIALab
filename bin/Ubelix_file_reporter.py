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


def feature_writer(path: str, feature_dic: dict, filename: str):
    if not os.path.exists(os.path.join(path, filename + '.txt')):

        with open(os.path.join(path, filename + '.txt'), 'w') as outfile:
            # I'm writing a header here just for the sake of readability
            # Any line starting with "#" will be ignored by numpy.loadtxt
            outfile.write('-' * 37 + '\n')
            outfile.write('-' * 10 + ' Features tested ' + '-' * 10)
            outfile.write('\n' + '-' * 37)
            outfile.write('\n')

            # Iterating through a ndimensional array produces slices along
            # the last axis. This is equivalent to data[i,:,:] in this case
            # for key, values in feature_dictionary.items():
            #         print(key, " : ", str(values))
            for key, values in feature_dic.items():
                # The formatting string indicates that I'm writing out
                # the values in left-justified columns 7 characters in width
                # with 2 decimal places.
                if isinstance(values, list):
                    featureL = str()
                    outfile.write('\n' + '# 1st order features ')
                    for features in values:
                        outfile.write('\n' + '  # ' + features)

                    outfile.write(featureL + '\n')

                else:
                    outfile.write('\n' + '# ' + values)


# feature_writer(path, featureList, filename)
