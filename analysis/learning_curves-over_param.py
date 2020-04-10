import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from itertools import tee
from src.analysis.learning_curve import save, plotBest
from src.experiment import ExperimentModel
from PyExpUtils.results.results import loadResults, whereParameterEquals, splitOverParameter
from PyExpUtils.utils.arrays import first
from PyExpUtils.utils.path import fileName, up

PARAM = 'alpha'

def getBest(results):
    best = first(results)

    for r in results:
        a = r.load()[0]
        b = best.load()[0]
        am = np.mean(a)
        bm = np.mean(b)
        if am > bm:
            best = r

    return best

def generatePlot(ax, exp_paths, bounds):
    for exp_path in exp_paths:
        exp = ExperimentModel.load(exp_path)

        results = loadResults(exp, 'return_summary.npy')
        results = whereParameterEquals(results, 'epsilon', 0.05)

        param_values = splitOverParameter(results, PARAM)

        for key, param_results in param_values.items():
            best = getBest(param_results)
            print('best parameters:', exp_path, f'{PARAM}={key}')
            print(best.params)

            alg = exp.agent

            b = plotBest(best, ax, label=alg + f' {PARAM}={key}', dashed=False)
            bounds.append(b)


if __name__ == "__main__":
    f, axes = plt.subplots(1)

    bounds = []

    exp_paths = sys.argv[1:]

    generatePlot(axes, exp_paths, bounds)

    plt.show()
    exit()

    save_path = 'experiments/exp/plots'
    os.makedirs(save_path, exist_ok=True)

    width = 8
    height = (24/5)
    f.set_size_inches((width, height), forward=False)
    plt.savefig(f'{save_path}/learning-curve.png', dpi=100)
