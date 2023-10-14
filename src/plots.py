import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def conf_plot(ground_truth, predictions, title):
    mean_prediciton = predictions.mean(axis=0)
    bins = pd.cut(mean_prediciton, bins=10, labels=[
                  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
    for bin in bins.unique():
        ax.scatter(int(bin)+0.5, ground_truth[bins == bin].mean(), color='k',)
        ax.arrow(0, 0, 10, 1, color='blue', head_length=0.0, head_width=0.0)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_xlabel('mean prediction')
        ax.set_ylabel('real fitness')
        ax.grid()
        # fig.savefig(f'output/{title}.png')
    return fig
