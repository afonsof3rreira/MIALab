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
    if not os.path.exists(os.path.join(path, filename + '.txt')):

        with open(os.path.join(path, filename + '.txt'), 'w') as outfile:

            outfile.write('-' * 37 + '\n')
            outfile.write('-' * 10 + ' Features tested ' + '-' * 10)
            outfile.write('\n' + '-' * 37)
            outfile.write('\n')

            for key, values in feature_dic.items():

                if isinstance(values, list):
                    featureL = str()
                    outfile.write('\n' + '# ' + key)
                    for features in values:
                        outfile.write('\n' + '  # ' + features)

                    outfile.write(featureL + '\n')

                else:
                    outfile.write('\n' + '# ' + key)

            print('\n' + 'Pipeline running time = ' + f'{round(running_time, 3)} second(s)')
# feature_writer(path, featureList, filename)
