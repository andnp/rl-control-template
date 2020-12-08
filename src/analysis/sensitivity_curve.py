import numpy as np
from itertools import tee

from PyExpUtils.results.results import sliceOverParameter, splitOverParameter, getBest

def getCurveReducer(bestBy: str):
    if bestBy == 'auc':
        return np.mean

    if bestBy == 'end':
        return lambda m: np.mean(m[-int(m.shape[0] * .1):])

def getSensitivityData(results, param, reducer='best', bestBy='auc'):
    if reducer == 'best':
        split = splitOverParameter(results, param)
        bestStream = {}
        for k in split:
            bestStream[k] = getBest(split[k], prefer='big')

    elif reducer == 'slice':
        l, r = tee(results)
        best = getBest(l, prefer='big')

        bestStream = sliceOverParameter(r, best, param)

    else:
        raise NotImplementedError()

    x = sorted(list(bestStream))
    best = bestStream


    metric = getCurveReducer(bestBy)

    y = np.array([metric(best[k].mean()) for k in x])
    e = np.array([metric(best[k].stderr()) for k in x])

    e[np.isnan(y)] = 0.000001
    y[np.isnan(y)] = 100000

    return x, y, e

def plotSensitivity(results, param, ax, reducer='best', stderr=True, color=None, label=None, dashed=False, dotted=False, bestBy='auc'):
    x, y, e = getSensitivityData(results, param, reducer=reducer, bestBy=bestBy)

    if dashed:
        dashes = '--'
    elif dotted:
        dashes = ':'
    else:
        dashes = None

    pl = ax.plot(x, y, label=label, linestyle=dashes, color=color, linewidth=2)
    if stderr:
        color = color if color is not None else pl[0].get_color()
        low_ci, high_ci = confidenceInterval(np.array(y), np.array(e))
        ax.fill_between(x, low_ci, high_ci, color=color, alpha=0.3)

def confidenceInterval(mean, stderr):
    return (mean - 2.0 * stderr, mean + 2.0 * stderr)
