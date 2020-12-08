import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from RlGlue import RlGlue
from src.experiment import ExperimentModel
from src.problems.registry import getProblem
from src.utils.Collector import Collector
from src.utils.rlglue import OneStepWrapper

if len(sys.argv) < 3:
    print('run again with:')
    print('python3 src/main.py <runs> <path/to/description.json> <idx>')
    exit(1)

runs = int(sys.argv[1])
exp = ExperimentModel.load(sys.argv[2])
idx = int(sys.argv[3])

max_steps = exp.max_steps

collector = Collector()
broke = False
for run in range(runs):
    # set random seeds accordingly
    np.random.seed(run)

    inner_idx = exp.numPermutations() * run + idx
    Problem = getProblem(exp.problem)
    problem = Problem(exp, inner_idx)

    agent = problem.getAgent()
    env = problem.getEnvironment()

    wrapper = OneStepWrapper(agent, problem.getGamma(), problem.rep)

    glue = RlGlue(wrapper, env)

    # Run the experiment
    rewards = []
    for episode in range(exp.episodes):
        glue.total_reward = 0
        glue.runEpisode(max_steps)

        # if the weights diverge to nan, just quit. This run doesn't matter to me anyways now.
        if np.isnan(np.sum(agent.w)):
            collector.fillRest(np.nan, exp.episodes)
            broke = True
            break

        collector.collect('return', glue.total_reward)

    collector.reset()
    if broke:
        break


# import matplotlib.pyplot as plt
# from src.utils.plotting import plot
# fig, ax1 = plt.subplots(1)

# return_data = collector.getStats('return')
# plot(ax1, return_data)
# ax1.set_title('Return')

# plt.show()
# exit()

from PyExpUtils.results.backends.csv import saveResults
from PyExpUtils.utils.arrays import downsample

for key in collector.all_data:
    data = collector.all_data[key]
    for run, datum in enumerate(data):
        inner_idx = exp.numPermutations() * run + idx

        # heavily downsample the data to reduce storage costs
        # we don't need all of the data-points for plotting anyways
        # method='window' returns a window average
        # method='subsample' returns evenly spaced samples from array
        # num=1000 makes sure final array is of length 1000
        # percent=0.1 makes sure final array is 10% of the original length (only one of `num` or `percent` can be specified)
        datum = downsample(datum, num=500, method='window')

        saveResults(exp, inner_idx, key, datum, precision=2)
