import matplotlib.pyplot as plt
import numpy as np


def main():

    # todo: load the "results.csv" file from the mia-results directory

    df = pd.read_csv("results.csv", delimiter=';')

    # todo: read the data into a list

    Amygdala_Dice, GreyMatter_Dice, Hippocampus_Dice, Thalamus_Dice, WhiteMatter_Dice = list(), list(), list(), list(), list()

    # iterating and assigning values for each label

    i = 0
    while i < len(df.values):

        if (i % 5) == 0:
            Amygdala_Dice.append(df.values[i][2])

        if (i % 6) == 0 and i != 0:
            GreyMatter_Dice.append(df.values[i][2])

        if (i % 7) == 0 and i != 0:
            Hippocampus_Dice.append(df.values[i][2])

        if (i % 8) == 0 and i != 0:
            Thalamus_Dice.append(df.values[i][2])

        if (i % 9) == 0 and i != 0:
            WhiteMatter_Dice.append(df.values[i][2])

        i += 1

    # todo: plot the Dice coefficients per label (i.e. white matter, gray matter, hippocampus, amygdala, thalamus) in a boxplot

    # data as a numpy array

    data = [np.asarray(Amygdala_Dice, float), np.asarray(GreyMatter_Dice, float), np.asarray(Hippocampus_Dice, float),
            np.asarray(Thalamus_Dice, float), np.asarray(WhiteMatter_Dice, float)]
    fig, ax = plt.subplots()

    ax.boxplot(data)

    # title, grid and labels

    ax.set_title('Dice Coefficients')
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    plt.xticks([1, 2, 3, 4, 5], ['Amygdala', 'GreyMatter', 'Hippocampus', 'Thalamus', 'WhiteMatter'], size=9)

    plt.show()

    # alternative: instead of manually loading/reading the csv file you could also use the pandas package
    # but you will need to install it first ('pip install pandas') and import it to this file ('import pandas as pd')
    pass  # pass is just a placeholder if there is no other code


if __name__ == '__main__':
    main()
