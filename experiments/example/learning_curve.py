import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.getcwd() + '/src')

from PyExpPlotting.learning_curves import plotBest
from PyExpPlotting.tools import findExperiments
from PyExpPlotting.matplot import save, setDefaultConference

from PyExpUtils.utils.dict import get
from PyExpUtils.results.results import loadResults
from analysis.results import findExpPath
from analysis.colors import basicControlColors
from experiment.tools import parseCmdLineArgs
from experiment import ExperimentModel

# makes sure figures are right size for the paper/column widths
# also sets fonts to be right size when saving
setDefaultConference('jmlr')

def whereParameterEquals(results, param, value):
    return filter(lambda r: get(r.params, param, value) == value, results)

ALG_ORDER = ['DQN', 'EQRC', 'ESARSA']

def generatePlot(ax, exp_paths):
    for alg in ALG_ORDER:
        exp_path = findExpPath(exp_paths, alg)

        exp = ExperimentModel.load(exp_path)
        results = loadResults(exp, 'step_return.h5')

        plotBest(results, ax, {
            'color': basicControlColors.get(alg),
            'label': alg,
            'prefer': 'big',
        })


if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()

    exps = findExperiments(key='{domain}')
    for domain in exps:
        print(domain)

        f, axes = plt.subplots(1)
        generatePlot(axes, exps[domain])

        if should_save:
            save(
                save_path=f'{path}/plots',
                plot_name=domain,
                save_type=save_type,
                width=1,
                f=f,
            )

        else:
            plt.show()
            exit()
