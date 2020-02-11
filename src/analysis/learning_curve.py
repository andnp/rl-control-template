import os
import numpy as np
from PyExpUtils.utils.generator import group
from PyExpUtils.results.results import getBest
import matplotlib.pyplot as plt

def windowAverage(arr, window):
    for g in group(arr, window):
        yield np.mean(g)

def smoothingAverage(arr, p=0.5):
    m = 0
    for i, a in enumerate(arr):
        if i == 0:
            m = a
            yield m
        else:
            m = p * m + (1 - p) * a
            yield m

def confidenceInterval(mean, stderr):
    return (mean - stderr, mean + stderr)

def save(exp, name, type='pdf'):
    exp_name = exp.getExperimentName()
    save_path = f'experiments/{exp_name}/plots'
    os.makedirs(save_path, exist_ok=True)
    final_path = f'{save_path}/{name}.{type}'
    plt.savefig(final_path)
    return final_path

def plot(results, ax, color=None, label=None, labelParams=None, bestBy='end', dashed=False):
    if bestBy == 'end':
        best = getBest(results, percent=0.1)
    elif bestBy == 'auc':
        best = getBest(results)
    else:
        raise Exception('I can only get best by "end" or "auc"')

    print(best.exp.agent, best.params)
    return plotBest(best, ax, color, label, labelParams=labelParams, dashed=dashed)


def getMaxY(arr):
    m = arr[0]
    for y in arr:
        if np.isnan(y) or np.isinf(y):
            continue

        if y > 100 * m:
            return np.nan

        if y > m:
            m = y

    return m

def plotBest(best, ax, window=1, smoothing=0, color=None, label=None, alpha=0.4, alphaMain=1, stderr=True, labelParams=None, dashed=False):
    label = label if label is not None else best.exp.agent

    params = ''
    if labelParams is not None:
        l = [f'{key}-{best.params[key]}' for key in labelParams]
        params = ' ' + ' '.join(l)

    mean = best.mean()
    ste = best.stderr()

    if len(mean.shape) == 1:
        mean = np.reshape(mean, (-1, 1))
        ste = np.reshape(ste, (-1, 1))

    if type(label) != list:
        label = [label] * mean.shape[1]

    if type(dashed) != list:
        dashed = [dashed] * mean.shape[1]

    for i in range(mean.shape[1]):
        lineplot(ax, mean[:, i], stderr=ste[:, i], smoothing=smoothing, window=window, color=color, label=label[i] + params, alpha=alpha, alphaMain=alphaMain, dashed=dashed[i])

    if len(mean.shape) > 1 and mean.shape[1] > 1:
        return (np.nan, np.nan)

    max_y = getMaxY(mean) * 1.05
    min_y = min(mean) * .95

    return (min_y, max_y)

def lineplot(ax, mean, window=1, smoothing=0, stderr=None, color=None, label=None, alpha=0.4, alphaMain=1, dashed=None):
    if dashed:
        dashes = ':'
    else:
        dashes = None

    if window > 1:
        mean = windowAverage(mean, window)

    if window > 1 and stderr is not None:
        stderr = windowAverage(stderr, window)

    mean = np.array(list(smoothingAverage(mean, smoothing)))

    ax.plot(mean, linestyle=dashes, label=label, color=color, alpha=alphaMain, linewidth=2)
    if stderr is not None:
        stderr = np.array(list(smoothingAverage(stderr, smoothing)))
        (low_ci, high_ci) = confidenceInterval(mean, stderr)
        ax.fill_between(range(mean.shape[0]), low_ci, high_ci, color=color, alpha=alpha * alphaMain)

    ax.legend()
