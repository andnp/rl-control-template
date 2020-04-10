import time
import sys
import os
sys.path.append(os.getcwd())

import src.experiment.ExperimentModel as Experiment
import PyExpUtils.runner.Slurm as Slurm
from PyExpUtils.results.paths import listResultsPaths
from PyExpUtils.utils.generator import group

if len(sys.argv) < 4:
    print('Please run again using')
    print('python -m scripts.scriptName [path/to/slurm-def] [src/executable.py] [base_path] [runs] [paths/to/descriptions]...')
    exit(0)

# -------------------------------
# Generate scheduling bash script
# -------------------------------

# the contents of the string below will be the bash script that is scheduled on compute canada
# change the script accordingly (e.g. add the necessary `module load X` commands)
cwd = os.getcwd()
def getJobScript(parallel):
    return f"""#!/bin/bash
cd {cwd}
. env/bin/activate
{parallel}
    """

# --------------------------
# Get command-line arguments
# --------------------------
slurm_path = sys.argv[1]
executable = sys.argv[2]
base_path = sys.argv[3]
runs = int(sys.argv[4])
experiment_paths = sys.argv[5:]

# generates a list of indices whose results are missing
def generateMissing(paths):
    for i, p in enumerate(paths):
        summary_path = p + '/return_summary.npy' # <-- TODO: change this to match the filename where you save your results
        if not os.path.exists(summary_path):
            yield i

# prints a progress bar
def printProgress(size, it):
    for i, _ in enumerate(it):
        print(f'{i + 1}/{size}', end='\r')
        if i - 1 == size:
            print()
        yield _

# ----------------
# Scheduling logic
# ----------------
for path in experiment_paths:
    print(path)
    # load the experiment json file
    exp = Experiment.load(path)
    # load the slurm config file
    slurm = Slurm.fromFile(slurm_path)

    # figure out how many indices to use
    size = exp.numPermutations()

    # get a list of all expected results paths
    paths = listResultsPaths(exp, runs)
    paths = printProgress(size, paths)
    # get a list of the indices whose results paths are missing
    indices = generateMissing(paths)

    # compute how many "tasks" to clump into each job
    groupSize = slurm.tasks * slurm.tasksPerNode

    for g in group(indices, groupSize):
        l = list(g)
        print("scheduling:", path, l)

        # build the executable string
        runner = f'python {executable} {runs} {path} '
        # generate the gnu-parallel command for dispatching to many CPUs across server nodes
        parallel = Slurm.buildParallel(runner, l, {
            'ntasks': slurm.tasks,
            'nodes-per-process': 1,
            'threads-per-process': 1, # <-- if you need multiple threads for a single process, change this number (NOTE: the rest of the scheduling logic does not know how to handle this yet)
        })

        # generate the bash script which will be scheduled
        script = getJobScript(parallel)

        ## uncomment for debugging the scheduler to see what bash script would have been scheduled
        # print(script)
        # exit()

        # make sure to only request the number of CPU cores necessary
        slurm.tasks = min([slurm.tasks, len(l)])
        Slurm.schedule(script, slurm)

        # DO NOT REMOVE. This will prevent you from overburdening the slurm scheduler. Be a good citizen.
        time.sleep(2)
