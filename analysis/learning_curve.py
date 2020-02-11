import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.learning_curve import plot, save, plotBest
from src.experiment import ExperimentModel
from PyExpUtils.results.results import loadResults, whereParameterGreaterEq, whereParameterEquals, find, getBest
from PyExpUtils.utils.arrays import first
from PyExpUtils.utils.path import fileName, up

error = 'rmsve'
problem = 'ShortChainInverted4060'
algorithm = 'td'

def generatePlot(ax, exp_paths, bounds):
    for exp_path in exp_paths:
        exp = ExperimentModel.load(exp_path)
        raise Exception('Set the name of the results file saved from these experiments')

        results = loadResults(exp, 'results.npy')
        results = whereParameterEquals(results, 'initial_value', 0)

        best = getBest(results)

        b = plotBest(best, ax, label='TD', color='yellow', dashed=False)
        bounds.append(b)


if __name__ == "__main__":
    f, axes = plt.subplots(1)

    bounds = []

    exp_paths = glob.glob(f'experiments/exp/{problem}/{algorithm}/*.json')

    generatePlot(axes, exp_paths, bounds)

    axes.set_title(f'{problem}')

    # axes[i, j].set_ylim([lower, upper])

    plt.show()
    exit()

    save_path = 'experiments/exp/plots'
    os.makedirs(save_path, exist_ok=True)

    width = 8
    height = (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/{algorithm}_{error}_learning-curves.png', dpi=100)
