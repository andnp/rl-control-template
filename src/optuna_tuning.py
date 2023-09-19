import Box2D     # we need to import this first because cedar is stupid
import os
import sys
sys.path.append(os.getcwd())

import socket
import logging
import argparse
import numpy as np
from functools import partial
from multiprocessing.pool import Pool
from RlGlue import RlGlue
from experiment import OptunaExperiment
from problems.registry import getProblem
from utils.iterators import partition
from PyExpUtils.results.sqlite import saveCollector
from PyExpUtils.collection.Collector import Collector
from PyExpUtils.collection.Sampler import Ignore, Identity
from PyExpUtils.utils.generator import group

# ------------------
# -- Command Args --
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exp', type=str, required=True)
parser.add_argument('-i', '--idx', type=int, required=True)
parser.add_argument('--save_path', type=str, default='./')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/')
parser.add_argument('--silent', action='store_true', default=False)
parser.add_argument('--gpu', action='store_true', default=False)
parser.add_argument('-j', '--jobs', type=int, default=5)

args = parser.parse_args()

# ---------------------------
# -- Library Configuration --
# ---------------------------
import jax
device = 'gpu' if args.gpu else 'cpu'
jax.config.update('jax_platform_name', device)

logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('filelock').setLevel(logging.ERROR)
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('jax').setLevel(logging.WARNING)
logging.getLogger('optuna').setLevel(logging.WARNING)
logger = logging.getLogger('exp')
prod = 'cdr' in socket.gethostname() or args.silent
if not prod:
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)


# --------------------------
# -- Single run execution --
# --------------------------
def execute(exp: OptunaExperiment.ExperimentModel, config_id: int, internal_seed: int):
    Problem = getProblem(exp.problem)

    collector = Collector(
        # specify which keys to actually store and ultimately save
        # Options are:
        #  - Identity() (save everything)
        #  - Window(n)  take a window average of size n
        #  - Subsample(n) save one of every n elements
        config={
            'return': Identity(),
            'episode': Identity(),
            'steps': Identity(),
        },
        # by default, ignore keys that are not explicitly listed above
        default=Ignore(),
    )
    collector.setIdx(config_id)
    collector.setContext(exp.get_flat_hypers(config_id))
    collector.addContext('config', config_id)
    collector.addContext('seed', exp.run)
    collector.addContext('internal_seed', internal_seed)

    # set random seeds accordingly
    seed = exp.run * exp.evaluation_runs + internal_seed
    np.random.seed(seed)

    # build stateful things and attach to checkpoint
    problem = Problem(exp, config_id, collector)
    problem.seed = seed
    agent = problem.getAgent()
    env = problem.getEnvironment()

    glue = RlGlue(agent, env)
    episode = 0

    glue.start()
    score = 0.
    for step in range(exp.evaluation_steps):
        collector.next_frame()
        interaction = glue.step()

        if interaction.t or (exp.episode_cutoff > -1 and glue.num_steps >= exp.episode_cutoff):
            # allow agent to cleanup traces or other stateful episodic info
            agent.cleanup()

            # collect some data
            score += (glue.total_reward * (glue.num_steps / exp.evaluation_steps))
            collector.collect('return', glue.total_reward)
            collector.collect('episode', episode)
            collector.collect('steps', glue.num_steps)

            # track how many episodes are completed (cutoff is counted as termination for this count)
            episode += 1
            glue.start()

    # if an episode is in-progress, go ahead and store
    # its current reward as well
    if glue.num_steps > 0:
        score += (glue.total_reward * (glue.num_steps / exp.evaluation_steps))

    collector.reset()

    # ------------
    # -- Saving --
    # ------------
    saveCollector(exp, collector, base=args.save_path)

    return float(score)

def setup(exp: OptunaExperiment.ExperimentModel, config_id: int):
    for seed in range(exp.evaluation_runs):
        exp.next_hypers(config_id)
        yield config_id, seed

def flatten(it):
    return (el for iter in it for el in iter)

if __name__ == '__main__':
    pool = Pool(args.jobs)
    exp = OptunaExperiment.load(args.exp)

    exp.set_idx(args.idx)
    epochs = range(exp.search_epochs)
    setup_it = map(partial(setup, exp), epochs)
    groups = group(setup_it, exp.sim_epochs)

    for g in groups:
        tasks = flatten(g)
        config_ids, seeds = zip(*tasks)
        scores = pool.starmap(partial(execute, exp), zip(config_ids, seeds))

        score_map = partition(zip(config_ids, scores))
        for config_id, score in score_map.items():
            s = np.mean(score)
            exp.record_metric(config_id, float(s))

            logger.debug('-' * 30)
            logger.debug(f'{config_id}')
            logger.debug(exp.get_flat_hypers(config_id))
            logger.debug(score)
