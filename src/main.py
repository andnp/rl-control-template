import Box2D # we need to import this first because cedar is stupid
import numpy as np
import logging
import socket
import time
import sys
import os
sys.path.append(os.getcwd())

from RlGlue import RlGlue
from PyExpUtils.utils.Collector import Collector
from experiment import ExperimentModel
from problems.registry import getProblem
from utils.rlglue import OneStepWrapper

# ---------------------------
# -- Library Configuration --
# ---------------------------
import jax
jax.config.update('jax_platform_name', 'cpu')

logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('filelock').setLevel(logging.ERROR)
logging.getLogger('numba').setLevel(logging.WARNING)
prod = len(sys.argv) == 4 or 'cdr' in socket.gethostname()
# prod = True
if not prod:
    logging.basicConfig(level=logging.DEBUG)

# ------------------
# -- Command Args --
# ------------------

if len(sys.argv) < 3:
    print('run again with:')
    print('python3 src/main.py <path/to/description.json> <indices ...>')
    exit(1)

# ----------------------
# -- Experiment Def'n --
# ----------------------

exp = ExperimentModel.load(sys.argv[1])
indices = list(map(int, sys.argv[2:]))

Problem = getProblem(exp.problem)
collector = Collector()
for idx in indices:
    collector.setIdx(idx)
    run = exp.getRun(idx)

    # set random seeds accordingly
    np.random.seed(run)

    problem = Problem(exp, idx, collector)
    agent = problem.getAgent()
    env = problem.getEnvironment()

    wrapper = OneStepWrapper(agent, problem.gamma, problem.rep)
    glue = RlGlue(wrapper, env)

    # Run the experiment
    glue.start()
    start_time = time.time()

    episode = 0
    for step in range(exp.total_steps):
        _, _, _, t = glue.step()

        if t or (exp.episode_cutoff > -1 and glue.num_steps >= exp.episode_cutoff):
            # track how many episodes are completed (cutoff is counted as termination for this count)
            episode += 1

            # allow agent to cleanup traces or other stateful episodic info
            agent.cleanup()

            # collect some data
            collector.concat('step_return', [glue.total_reward] * glue.num_steps)
            collector.collect('episodic_return', glue.total_reward)

            # compute the average time-per-step in ms
            avg_time = 1000 * (time.time() - start_time) / step
            logging.debug(f' {episode} {step} {glue.total_reward} {avg_time:.4}ms')

            glue.start()

    # try to detect if a run never finished
    # if we have no data in the 'step_return' key, then the termination condition was never hit
    if len(collector.get('step_return', idx)) == 0:
        # collect an array of rewards that is the length of the number of steps in episode
        # effectively we count the whole episode as having received the same final reward
        collector.concat('step_return', [glue.total_reward] * glue.num_steps)
        # also track the reward per episode (this may not have the same length for all agents!)
        collector.collect('episodic_return', glue.total_reward)

    # force the data to always have same length
    collector.fillRest('step_return', exp.total_steps)

# -------------------------
# -- [Optional] Plotting --
# -------------------------

# import matplotlib.pyplot as plt
# from src.utils.plotting import plot
# fig, ax1 = plt.subplots(1)

# return_data = collector.getStats('return')
# plot(ax1, return_data)
# ax1.set_title('Return')

# plt.show()
# exit()

# ------------
# -- Saving --
# ------------

from PyExpUtils.results.backends.pandas import saveCollector

for key in collector.keys():
    # heavily downsample the data to reduce storage costs
    # we don't need all of the data-points for plotting anyways
    # method='window' returns a window average
    # method='subsample' returns evenly spaced samples from array
    # num=1000 makes sure final array is of length 1000
    # percent=0.1 makes sure final array is 10% of the original length (only one of `num` or `percent` can be specified)

    # don't downsample episode returns
    if 'episodic' not in key:
        collector.downsample(key, num=500, method='window')

saveCollector(exp, collector)
