import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from RlGlue import RlGlue
from src.experiment import ExperimentModel
from src.problems.registry import getProblem
from src.utils.Collector import Collector
from src.utils.rlglue import OneStepWrapper

from RlGlue.environment import BaseEnvironment
from RlGlue.agent import BaseAgent

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

    Problem = getProblem(exp.problem)
    problem = Problem(exp, idx)

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

return_data = collector.getStats('return')

# import matplotlib.pyplot as plt
# from src.utils.plotting import plot
# fig, ax1 = plt.subplots(1)

# plot(ax1, return_data)
# ax1.set_title('Return')

# plt.show()
# exit()

# save results to disk
save_context = exp.buildSaveContext(idx, base="./")
save_context.ensureExists()

np.save(save_context.resolve('return_summary.npy'), return_data)