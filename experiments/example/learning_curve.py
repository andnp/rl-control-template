import os
import sys
sys.path.append(os.getcwd() + '/src')

import matplotlib.pyplot as plt
import numpy as np
from experiment.tools import parseCmdLineArgs
from experiment.ExperimentModel import ExperimentModel
from utils.results import ResultCollection


from PyExpPlotting.matplot import save, setDefaultConference
import rlevaluation.hypers as Hypers
from rlevaluation.statistics import Statistic
from rlevaluation.temporal import TimeSummary, extract_learning_curves, curve_percentile_bootstrap_ci
from rlevaluation.config import data_definition
from rlevaluation.interpolation import compute_step_return

setDefaultConference('jmlr')


COLORS = {
    'DQN': 'tab:blue',
    'EQRC': 'purple',
}


if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()

    results = ResultCollection(Model=ExperimentModel)
    data_definition(
        hyper_cols=results.get_hyperparameter_columns(),
        seed_col='seed',
        time_col='frame',
        environment_col=None,
        algorithm_col=None,
        make_global=True,
    )

    for env, sub_results in results.groupby_directory(level=2):
        fig, ax = plt.subplots(1, 1)
        for alg_result in sub_results:
            alg = alg_result.filename

            df = alg_result.load()
            if df is None:
                continue

            report = Hypers.select_best_hypers(
                df,
                metric='return',
                prefer=Hypers.Preference.low,
                time_summary=TimeSummary.mean,
                statistic=Statistic.mean,
            )

            exp = alg_result.exp

            xs, ys = extract_learning_curves(
                df,
                hyper_vals=report.best_configuration,
                metric='return',
                interpolation=lambda x, y: compute_step_return(x, y, exp.total_steps),
            )

            xs = np.asarray(xs)[:, ::exp.total_steps // 1000]
            ys = np.asarray(ys)[:, ::exp.total_steps // 1000]
            assert np.all(np.isclose(xs[0], xs))

            res = curve_percentile_bootstrap_ci(
                rng=np.random.default_rng(0),
                y=ys,
                statistic=Statistic.mean,
                iterations=10000,
            )

            ax.plot(xs[0], res.sample_stat, label=alg, color=COLORS[alg], linewidth=1.0)
            ax.fill_between(xs[0], res.ci[0], res.ci[1], color=COLORS[alg], alpha=0.2)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        path = os.path.sep.join(os.path.relpath(__file__).split(os.path.sep)[:-1])
        save(
            save_path=f'{path}/plots',
            plot_name=env,
            f=fig,
            height_ratio=2/3,
        )
