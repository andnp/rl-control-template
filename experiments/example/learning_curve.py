import os
import sys
sys.path.append(os.getcwd() + '/src')

import numpy as np
import matplotlib.pyplot as plt
from PyExpPlotting.matplot import save, setDefaultConference
from PyExpUtils.results.Collection import ResultCollection
from PyExpUtils.results.tools import collapseRuns
from RlEvaluation.intervals import bootstrap

# from analysis.confidence_intervals import bootstrapCI
from experiment.ExperimentModel import ExperimentModel
from experiment.tools import parseCmdLineArgs

# makes sure figures are right size for the paper/column widths
# also sets fonts to be right size when saving
setDefaultConference('jmlr')

COLORS = {
    'EQRC': 'blue',
    'ESARSA': 'red',
    'DQN': 'black',
    'PrioritizedDQN': 'purple',
}

if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()

    collection = ResultCollection.fromExperiments('step_return', Model=ExperimentModel)
    collection.apply(collapseRuns)

    for env in collection.keys():
        print(env)
        algs = list(collection[env])
        f, ax = plt.subplots()
        for alg in collection[env]:
            df = collection[env, alg]
            if df is None:
                continue

            best_idx = df['data'].apply(np.mean).argmax()
            best = df.iloc[best_idx]
            print('-' * 30)
            print(alg)
            print(best)

            line = bootstrap(best, column='data', bootstraps=1000)

            lo = line[0]
            avg = line[1]
            hi = line[2]
            ax.plot(avg, label=alg, color=COLORS[alg], linewidth=0.25)
            ax.fill_between(range(line.shape[1]), lo, hi, color=COLORS[alg], alpha=0.2)

        ax.legend()

        if should_save:
            save(
                save_path=f'{path}/plots',
                plot_name=f'{env}'
            )
            plt.clf()
        else:
            plt.show()
            exit()
