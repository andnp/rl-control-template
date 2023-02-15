import Box2D     # we need to import this first because cedar is stupid
import numpy as np
import logging
import socket
import time
import sys
import os
sys.path.append(os.getcwd())

from RlGlue import RlGlue
from PyExpUtils.utils.Collector import Collector
from PyExpUtils.results.backends.pandas import saveCollector
from experiment import ExperimentModel
from problems.registry import getProblem
from utils.checkpoint import Checkpoint

# ---------------------------
# -- Library Configuration --
# ---------------------------
import jax
jax.config.update('jax_platform_name', 'cpu')

logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('filelock').setLevel(logging.ERROR)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('jax').setLevel(logging.WARNING)
logger = logging.getLogger('exp')
prod = 'cdr' in socket.gethostname()
# prod = True
if not prod:
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)

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
for idx in indices:
    chk = Checkpoint(exp, idx)
    chk.load_if_exists()

    collector = chk.build('collector', Collector)

    collector.setIdx(idx)
    run = exp.getRun(idx)

    # set random seeds accordingly
    np.random.seed(run)

    # build stateful things and attach to checkpoint
    problem = chk.build('p', lambda: Problem(exp, idx, collector))
    agent = chk.build('a', problem.getAgent)
    env = chk.build('e', problem.getEnvironment)

    glue = chk.build('glue', lambda: RlGlue(agent, env))
    chk['episode'] = 0

    # Run the experiment
    start_time = time.time()

    # if we haven't started yet, then make the first interaction
    if glue.total_steps == 0:
        glue.start()

    for step in range(glue.total_steps, exp.total_steps):
        chk.maybe_save()
        interaction = glue.step()

        if interaction.t or (exp.episode_cutoff > -1 and glue.num_steps >= exp.episode_cutoff):
            # track how many episodes are completed (cutoff is counted as termination for this count)
            chk['episode'] += 1

            # allow agent to cleanup traces or other stateful episodic info
            agent.cleanup()

            # collect some data
            collector.concat('step_return', [glue.total_reward] * glue.num_steps)
            collector.collect('episodic_return', glue.total_reward)

            # compute the average time-per-step in ms
            avg_time = 1000 * (time.time() - start_time) / (step + 1)
            fps = step / (time.time() - start_time)

            episode = chk['episode']
            logger.debug(f'{episode} {step} {glue.total_reward} {avg_time:.4}ms {int(fps)}')

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

    # ------------
    # -- Saving --
    # ------------

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
    chk.delete()
