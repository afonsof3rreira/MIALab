import argparse
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import re

matplotlib.use('Agg')


class selected_features:

    def __init__(self):
        self.fof_parameters = ['10Percentile',
                               '90Percentile',
                               'Energy',
                               'Entropy',
                               'InterquartileRange',
                               'Kurtosis',
                               'Maximum',
                               'MeanAbsoluteDeviation',
                               'Mean',
                               'Median',
                               'Minimum',
                               'Range',
                               'RobustMeanAbsoluteDeviation',
                               'RootMeanSquared',
                               'Skewness',
                               'TotalEnergy',
                               'Uniformity',
                               'Variance']

        self.glcm_parameters = ['Autocorrelation',
                                'ClusterProminence',
                                'ClusterShade',
                                'ClusterTendency',
                                'Contrast',
                                'Correlation',
                                'DifferenceAverage',
                                'DifferenceEntropy',
                                'DifferenceVariance',
                                'Id',
                                'Idm',
                                'Idmn',
                                'Idn',
                                'Imc1',
                                'Imc2',
                                'InverseVariance',
                                'JointAverage',
                                'JointEnergy',
                                'JointEntropy',
                                'MCC',
                                'MaximumProbability',
                                'SumAverage',
                                'SumEntropy',
                                'SumSquares']

    def GetFofList(self):
        return self.fof_parameters

    def GetSofList(self):
        return self.glcm_parameters


def boxplot(file_path: str, data: dict, title: str, used_metric: str, x_label: str, y_label: str,
            x_ticks: tuple, min_: float = None, max_: float = None):
    """Generates a boxplot for the chosen metric (y axis) comparing all the different tests for the chosen label (x-axis)

           Args:
               file_path (str): the output file path
               data (dict): the data containing DICE and HSDRF for each brain structure and each metric
               title (str): the plot title
               used_metric (str): the metric to be used
               x_label (str): the x-axis label (the chosen brain structure)
               y_label (str): the y-axis label
               x_ticks (tuple): the methods to be compared for each brain structure
               min_ (float): the bottom limit of the y-axis
               max_ (float): the top limit of the y-axis
    """
    # data = a nested dict
    # data = {'metric1': {'brain_struct1' : [array(vales for test 1), array(values for test2), ...],
    #                     'brain_struct2' : [array(vales for test 1), array(values for test2), ...],
    #                      ...
    #         'metric2': {'brain_struct1' : [array(vales for test 1), array(values for test2), ...],
    #                     'brain_struct2' : [array(vales for test 1), array(values for test2), ...],
    #                      ...
    #        }

    # adding the data from the chosen metric and label to a list
    concat_data = []
    test_len = None
    for key, metric in data.items():
        for sub_k, brain_structure in metric.items():
            if key == used_metric and sub_k == x_label:
                concat_data.extend(brain_structure)
                test_len = len(brain_structure)

    # the amount of data for each label has to be equal to the x_ticks length
    if test_len != len(x_ticks):
        raise ValueError('arguments data and x_ticks need to have compatible lengths')

    fig, ax = plt.subplots(figsize=(20, 10))
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    ax.boxplot(concat_data, vert=1, widths=0.6)

    # Add a horizontal grid to the plot, light in color
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                  alpha=0.5)

    # set and format title, labels, and ticks
    ax.set_title(title, fontweight='bold', fontsize=14, pad=20)
    ax.set_ylabel(y_label, fontweight='bold', fontsize=12, labelpad=20)

    # ax.set_xlabel(x_label, fontweight='bold', fontsize=9.5)
    ax.yaxis.set_tick_params(labelsize=12)

    # forming x_ticks
    x_tick_l = []
    for x in range(len(x_ticks)):
        if x_ticks[x] == 'a_baseline':  # in case of baseline
            x_tick_l.extend(['Baseline'])

        elif x_ticks[x] == 'MeanAbsoluteDeviation':  # in case of Mean Absolute Deviation
            x_tick_l.extend(['Mean A.D.'])

        elif x_ticks[x] == 'RobustMeanAbsoluteDeviation':  # in case of Robust Mean Absolute Deviation
            x_tick_l.extend(['Robust Mean A.D.'])

        elif x_ticks[x][0].isdigit():  # in case of 10percentile, ...

            # Separating numbers from letters
            tick = re.sub("[A-Za-z]+", lambda ele: " " + ele[0] + " ", x_ticks[x])
            ind_last_char = 0
            # adding th after percentile numbers
            for ind_char in range(len(tick)):
                if not tick[ind_char].isdigit():
                    ind_last_char = ind_char
                    break
            tick = tick[:ind_last_char] + 'th' + tick[ind_last_char:]

            # spacing words by uppercase
            tick = ''.join(' ' + char if char.isupper() else char.strip() for char in tick).strip()
            x_tick_l.extend(['{}'.format(tick)])

        # in case of 2 or more words merged, except if it's MMC
        elif sum(1 for c in x_ticks[x] if c.isupper()) > 1 and len(x_ticks[x]) > 3:
            tick = ''.join(' ' + char if char.isupper() else char.strip() for char in x_ticks[x]).strip()
            x_tick_l.extend(['{}'.format(tick)])

        else:  # otherwise (in case of having only one word in the string)
            x_tick_l.extend(['{}'.format(x_ticks[x])])

    ax.set_xticklabels(x_tick_l, fontdict={'fontsize': 8, 'fontweight': 'bold'}, rotation=35, fontsize=10,
                       linespacing=1.5)

    # remove frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # thicken frame
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Hide the grid behind plot objects
    ax.set(axisbelow=True)

    # adjust min and max if provided
    if min_ is not None or max_ is not None:
        min_original, max_original = ax.get_ylim()
        min_ = min_ if min_ is not None and min_ < min_original else min_original
        max_ = max_ if max_ is not None and max_ > max_original else max_original
        ax.set_ylim(min_, max_)

    plt.savefig(file_path)
    plt.close()


def format_data(data, label: str, metric: str):
    return data[data['LABEL'] == label][metric].values


def metric_to_readable_text(metric: str):
    if metric == 'DICE':
        return 'Dice coefficient [-]'
    elif metric == 'HDRFDST':
        return 'Hausdorff distance [mm]'
    else:
        raise ValueError('Metric "{}" unknown'.format(metric))


def metric_to_readable_text_title(metric: str):
    if metric == 'DICE':
        return 'Dice coefficient'
    elif metric == 'HDRFDST':
        return 'Hausdorff distance'
    else:
        raise ValueError('Metric "{}" unknown'.format(metric))


# this function was taken from slack https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python
# and used to order the crawled filenames in the preferred way
def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def read_data_feature_list(path_folder: str, result_filename='results.csv', include_baseline=False):
    """ Reads data from a folder containing a sub-folder for each test

           Args:
               path_folder (str): the folder from which to crawl the the results file
               result_filename (str): the name of the csv file inside each sub-folder
               include_baseline (bool): whether or not to include the baseline features

           Returns:
               dfs (list): a list containing the loaded values according to the sub-folder order
               methods (list): a list containing the names of the methods used (= the names of the sub-folders)
    """
    dfs_fof, dfs_sof = [], []
    methods_fof, methods_sof = [], []

    #   features
    features = selected_features()
    dirs_fof = features.GetFofList()
    dirs_sof = features.GetSofList()
    if include_baseline:
        dirs_fof.insert(0, 'a_baseline')
        dirs_sof.insert(0, 'a_baseline')

    # read through FOFs
    for feature_name in dirs_fof:
        dir_path = os.path.join(path_folder, feature_name)
        print(dir_path)
        methods_fof.append(feature_name)
        dfs_fof.append(pd.read_csv(os.path.join(dir_path, result_filename), sep=';'))
        print(os.path.join(dir_path, result_filename))

    # read through SOFs
    for feature_name in dirs_sof:
        dir_path = os.path.join(path_folder, feature_name)

        methods_sof.append(feature_name)
        dfs_sof.append(pd.read_csv(os.path.join(dir_path, result_filename), sep=';'))

    return dfs_fof, dfs_sof, methods_fof, methods_sof


def read_data_features(path_folder: str, result_filename='results.csv'):
    """ Reads data from a folder containing a subfolder for each feature (FOFs and GLCMs)

           Args:
               path_folder (str): the folder in which to crawl the results file
               result_filename (str): the name of the csv file inside each subfolder

           Returns:
               dfs (list): a list containing the loaded values according to the subfolder order
               methods (list): a list containing the names of the used methods (= the names of the subfolders)
    """

    dfs = []
    methods = []
    for root, dirs, _ in os.walk(path_folder, topdown=True):
        dirs = sorted_nicely(dirs)
        for dir_i in dirs:
            methods.append(dir_i)
            for sub_root, sub_dir, filenames in os.walk(os.path.join(root, dir_i)):
                for filename in filenames:
                    if filename == result_filename:
                        # print(os.path.join(sub_root, filename))
                        dfs.append(pd.read_csv(os.path.join(sub_root, filename), sep=';'))
    return dfs, methods


def main(path_folder, plot_dir: str):
    metrics = ('DICE', 'HDRFDST')  # the metrics we want to plot the results for

    metrics_yaxis_limits = ((0.0, 1.0), (0.0, None))

    labels = ('WhiteMatter', 'Amygdala', 'GreyMatter', 'Hippocampus',
              'Thalamus')  # the brain structures/tissues which we are plotting

    # load the CSVs
    #   TODO: change include_baseline to true to include its results in the boxplot
    dfs_fof, dfs_sof, methods_fof, methods_sof = read_data_feature_list(path_folder, include_baseline=True)

    # some parameters to improve the plot's readability
    methods_fof = tuple(methods_fof)
    methods_sof = tuple(methods_sof)

    title = '{} for {}'

    # loading data in a nested dictionary
    concat_data_fof = {}
    for metric in metrics:
        sub_dict = {}
        for label in labels:
            concat_data_fof.update({metric: {}})
            sub_dict.update({label: [format_data(df, label, metric) for df in dfs_fof]})

        concat_data_fof.update({metric: sub_dict})
    print(len(concat_data_fof['DICE']['WhiteMatter']))

    concat_data_sof = {}
    for metric in metrics:
        sub_dict = {}
        for label in labels:
            concat_data_sof.update({metric: {}})
            sub_dict.update({label: [format_data(df, label, metric) for df in dfs_sof]})

        concat_data_sof.update({metric: sub_dict})
    print(len(concat_data_sof['DICE']['WhiteMatter']))
    data = [concat_data_fof, concat_data_sof]
    methods = [methods_fof, methods_sof]
    methods_as_strings = ['FOF', 'SOF']
    print('-' * 10)
    for feature, method, method_str in zip(data, methods, methods_as_strings):
        for label in labels:
            for metric, (min_, max_) in zip(metrics, metrics_yaxis_limits):
                print(metric + ' ' + label)
                boxplot(os.path.join(plot_dir, '{}_{}_{}.png'.format(method_str, metric, label)),
                        feature,
                        title.format(metric_to_readable_text_title(metric), label),
                        metric,
                        label,
                        metric_to_readable_text(metric),
                        method,
                        min_, max_
                        )


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """
    parser = argparse.ArgumentParser(description='Result plotting.')

    parser.add_argument(
        '--path_folder',
        type=str,
        default='./mia-result',
        help='Path to the folder containing sub-folders that contain the result CSV files.'
    )

    parser.add_argument(
        '--plot_dir',
        type=str,
        default='./mia-result/2020-10-27-14-06-14',
        help='Path to the plot directory.'
    )

    args = parser.parse_args()
    main(args.path_folder, args.plot_dir)
