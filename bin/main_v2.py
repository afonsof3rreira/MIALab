from boxplot import main
import numpy as np

# method 1 directory, method 2 directory, output directory
main('./mia-result/2020-11-05-11-24-46/results.csv', './mia-result/2020-11-04-15-58-24/results.csv',
     './mia-result/2020-11-04-15-58-24')

vec = np.array([1, 2, 3, 4, 5, 6, 5.9, 6.1, 9.9, 10.2, 5, 4])
vec = np.sort(vec)
p_10 = np.percentile(vec, 10)
p_90 = np.percentile(vec, 90)


def function_s(array, min, max):
    """ 
    :param array: sorted 3D ndarray
    :param min: lower quantile value
    :param max: upper quantile value
    :return: ndarray containing values between min and max, inclusive
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
