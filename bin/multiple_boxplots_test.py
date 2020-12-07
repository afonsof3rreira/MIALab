import argparse
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import patches


def set_box_format(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['caps'], linewidth=0)
    plt.setp(bp['medians'], color=color)
    plt.setp(bp['medians'], linewidth=1.5)
    plt.setp(bp['fliers'], marker='.')
    plt.setp(bp['fliers'], markerfacecolor='black')
    plt.setp(bp['fliers'], alpha=1)


def boxplot1(file_path: str, data: dict, title: str, used_metric: str, x_label: str, y_label: str,
             x_ticks: tuple,
             min_: float = None, max_: float = None):
    # data = a nested dict
    # data = {'metric1': {'brain_struct1' : [array(vales for test 1), array(values for test2), ...],
    #                     'brain_struct2' : [array(vales for test 1), array(values for test2), ...],
    #                      ...
    #         'metric2': {'brain_struct1' : [array(vales for test 1), array(values for test2), ...],
    #                     'brain_struct2' : [array(vales for test 1), array(values for test2), ...],
    #                      ...
    #        }

    # x_label = metric

    concat_data = []
    # for sub_k, brain_structure in data[used_metric].items():
    #     concat_data.extend(brain_structure)

    test_len = None
    for key, metric in data.items():
        for sub_k, brain_structure in metric.items():
            if key == used_metric and sub_k == x_label:
                concat_data.extend(brain_structure)
                test_len = len(brain_structure)



    if test_len != len(x_ticks):
        raise ValueError('arguments data and x_ticks need to have compatible lengths')

    # fig = plt.figure(
    #     figsize=(1, 4.8))
    # figsize=(4.8 * 1.5, 6.4 * 1.5))  # figsize defaults to (width, height) =(6.4, 4.8),
    # for boxplots, we want the ratio to be inversed
    # ax = fig.add_subplot(111)  # create an axes instance (nrows=ncols=index)


    # for further latex interpreter usage (the latex.exe path needs to be added)
    # rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    # ## for Palatino and other serif fonts use:
    # # rc('font',**{'family':'serif','serif':['Palatino']})
    # rc('text', usetex=True)

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.canvas.set_window_title('A Boxplot Example')
    fig.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = ax.boxplot(concat_data, vert=1, widths=0.6)

    # Add a horizontal grid to the plot, light in color
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                  alpha=0.5)

    # set and format litle, labels, and ticks
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_ylabel(y_label, fontweight='bold', fontsize=12)

    ax.set_xlabel(x_label, fontweight='bold', fontsize=9.5)  # we don't use the x-label
    # since it should be clear from the x-ticks
    ax.yaxis.set_tick_params(labelsize=12)


    # brain_s_len = len(brain_structures) // test_len

    list = []
    for x in range(len(x_ticks)):
        list.extend(['{}'.format(x_ticks[x])])


    ax.set_xticklabels(list, fontdict={'fontsize': 8, 'fontweight': 'bold'}, rotation=35, fontsize=8, linespacing=1.5)

    # remove frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # thicken frame
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    ax.set(axisbelow=True)  # Hide the grid behind plot objects)
    # box_colors = ['darkkhaki', 'royalblue']
    #
    # num_boxes = len(brain_structures)
    # for i in range(num_boxes):
    #     box = bp['boxes'][i]
    #     box_x = []
    #     box_y = []
    #     for j in range(5):
    #         box_x.append(box.get_xdata()[j])
    #         box_y.append(box.get_ydata()[j])
    #     box_coords = np.column_stack([box_x, box_y])
    #     # Alternate between Dark Khaki and Royal Blue
    #     ax.add_patch(patches.Polygon(box_coords, facecolor=box_colors[i % 2]))

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


# def concat_formatted_data(data_list):


def metric_to_readable_text(metric: str):
    if metric == 'DICE':
        return 'Dice coefficient'
    elif metric == 'HDRFDST':
        return 'Hausdorff distance (mm)'
    else:
        raise ValueError('Metric "{}" unknown'.format(metric))


import os
from os import listdir
from os.path import isfile, join

import re


# this fucntion was taken from slack https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python

def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def main(path_folder, plot_dir: str):
    metrics = ('DICE', 'HDRFDST')  # the metrics we want to plot the results for
    # metrics_yaxis_limits = (
    # (0.0, 1.0), (0.0, None))  # tuples of y-axis limits (min, max) for each metric. Use None if unknown

    metrics_yaxis_limits = ((0.0, 1.0), (0.0, None))

    labels = ('WhiteMatter', 'Amygdala', 'GreyMatter', 'Hippocampus',
              'Thalamus')  # the brain structures/tissues you are interested in

    # load the CSVs. We usually want to compare different methods (e.g. a set of different features), therefore,
    # we load two CSV (for simplicity, it is the same here)
    dfs = []
    methods = []
    for root, dirs, _ in os.walk(path_folder, topdown=True):
        dirs = sorted_nicely(dirs)
        for dir_i in dirs:
            methods.append(dir_i)
            for sub_root, sub_dir, filenames in os.walk(os.path.join(root, dir_i)):
                for filename in filenames:
                    if filename == 'results.csv':
                        # print(os.path.join(sub_root, filename))
                        dfs.append(pd.read_csv(os.path.join(sub_root, filename), sep=';'))

    # df_method1 = pd.read_csv(csv_file, sep=';')
    # df_method2 = pd.read_csv(csv_file2, sep=';')
    #
    # dfs = [df_method1, df_method2]

    # some parameters to improve the plot's readability
    methods = tuple(methods)
    print(methods)
    # methods = ('Test 1', 'Test 3')
    title = 'Evaluation metrics for all RF parameters on {}'

    concat_data = {}
    for metric in metrics:
        sub_dict = {}
        for label in labels:
            concat_data.update({metric: {}})
            sub_dict.update({label: [format_data(df, label, metric) for df in dfs]})

        concat_data.update({metric: sub_dict})

    for label in labels:
        for metric, (min_, max_) in zip(metrics, metrics_yaxis_limits):
            print(metric + ' ' + label)
            sub_concat_data = concat_data[metric][label]
            boxplot1(os.path.join(plot_dir, '{}_{}.png'.format(metric, label)),
                     concat_data,
                     title.format('all brain structures'),
                     metric,
                     label,
                     metric_to_readable_text(metric),
                     methods,
                     min_, max_
                     )


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """
    parser = argparse.ArgumentParser(description='Result plotting.')

    parser.add_argument(
        '--csv_file',
        type=str,
        default='./mia-result/2020-10-27-14-06-14/results.csv',
        help='Path to the result CSV file.'
    )

    parser.add_argument(
        '--plot_dir',
        type=str,
        default='./mia-result/2020-10-27-14-06-14',
        help='Path to the plot directory.'
    )

    args = parser.parse_args()
    main(args.csv_file, args.plot_dir)
