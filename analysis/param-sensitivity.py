import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.sensitivity_curve import plotSensitivity
from src.analysis.colors import colors
from src.experiment import ExperimentModel
from PyExpUtils.results.results import loadResults, whereParameterEquals, splitOverParameter
from PyExpUtils.utils.arrays import first
from PyExpUtils.utils.path import fileName, up

param = 'alpha'

def generatePlot(ax, exp_paths, bounds):
    for exp_path in exp_paths:
        exp = ExperimentModel.load(exp_path)

        results = loadResults(exp, 'return_summary.npy')

        plotSensitivity(results, param, ax, label=exp.agent)


if __name__ == "__main__":
    f, axes = plt.subplots(1)

    bounds = []

    exp_paths = sys.argv[1:]

    generatePlot(axes, exp_paths, bounds)

    plt.legend()

    plt.show()
    exit()

    save_path = 'experiments/exp/plots'
    os.makedirs(save_path, exist_ok=True)

    width = 8
    height = (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/{param}-sensitivity.png', dpi=100)
