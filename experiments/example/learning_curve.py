import os
import sys
sys.path.append(os.getcwd() + '/src')

import matplotlib.pyplot as plt
from PyExpPlotting.matplot import save, setDefaultConference
from PyExpUtils.results.Collection import ResultCollection

import RlEvaluation.temporal as temporal
from RlEvaluation.ResultData import ResultData, Metric
from RlEvaluation.hypers import Preference

# from analysis.confidence_intervals import bootstrapCI
from experiment.ExperimentModel import ExperimentModel
from experiment.tools import parseCmdLineArgs

# makes sure figures are right size for the paper/column widths
# also sets fonts to be right size when saving
setDefaultConference('jmlr')

COLORS = {
    'AC_TC': 'grey',
    'EQRC': 'blue',
    'ESARSA': 'red',
    'DQN': 'black',
    'PrioritizedDQN': 'purple',
}

if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()

    collection = ResultCollection.fromExperiments(
        metrics=('step_return',),
        Model=ExperimentModel,
    )

    for env in collection.keys():
        print(env)
        algs = list(collection[env])
        f, ax = plt.subplots()

        all_empty = True
        for alg in collection[env]:
            df = collection[env, alg]
            if df is None:
                continue

            res = ResultData(
                data=df,
                config={
                    'step_return': Metric(time_summary=temporal.mean, preference=Preference.high),
                },
            )

            all_empty = False
            best_idx = res.get_best_hyper_idx('step_return')
            line = res.get_learning_curve('step_return', best_idx)

            lo = line[0]
            avg = line[1]
            hi = line[2]
            ax.plot(avg, label=alg, color=COLORS[alg], linewidth=0.25)
            ax.fill_between(range(line.shape[1]), lo, hi, color=COLORS[alg], alpha=0.2)

        if all_empty:
            continue

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
