import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from RlGlue import RlGlue
from src.experiment import ExperimentModel

from RlGlue.environment import BaseEnvironment
from RlGlue.agent import BaseAgent

if len(sys.argv) < 3:
    print('run again with:')
    print('python3 src/main.py <path/to/description.json> <idx>')
    exit(1)

max_steps = 100

exp = ExperimentModel.load()
idx = int(sys.argv[2])

# figure out which run number of these parameter settings this is
run = exp.getRun(idx)

# set random seeds accordingly
np.random.seed(run)

# TODO: replace with real agent class and environment class
glue = RlGlue(BaseAgent, BaseEnvironment)

# Run the experiment
rewards = []
glue.start()
for step in range(max_steps):
    # call agent.step and environment.step
    r, o, a, t = glue.step()
    
    # collect data throughout run
    rewards.append(r)

    # if terminal state, then restart the interface
    if t:
        glue.start()


# save results to disk
save_context = exp.buildSaveContext(idx, base="results")
save_context.ensureExists()

np.save(save_context.resolve('rewards.npy'), rewards)