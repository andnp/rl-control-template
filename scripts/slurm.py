import time
import sys
import os
sys.path.append(os.getcwd() + '/src')

import experiment.ExperimentModel as Experiment
import PyExpUtils.runner.Slurm as Slurm
from PyExpUtils.results.indices import listIndices
from PyExpUtils.results.paths import listResultsPaths
from PyExpUtils.utils.arrays import first
from PyExpUtils.utils.generator import group
from PyExpUtils.utils.csv import buildCsvParams

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
. ~/env/bin/activate

export MPLBACKEND=TKAgg
export OMP_NUM_THREADS=1
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
def generateMissing(exp, indices, data):
    for idx in indices:
        csv_params = buildCsvParams(exp, idx)
        run = exp.getRun(idx)
        key = f'{csv_params},{run}'
        if not any(s.startswith(key) for s in data):
            yield idx


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
    size = exp.numPermutations() * runs

    paths = listResultsPaths(exp, runs)
    res_path = first(paths)

    data = []
    raise NotImplementedError("Don't forget to change the expected result file")
    data_path = f'{res_path}/TODO-CHANGE-ME.csv'
    if os.path.exists(data_path):
        f = open(data_path, 'r')
        data = f.readlines()
        f.close()

    indices = listIndices(exp, runs)
    # get all of the indices corresponding to missing results
    indices = generateMissing(exp, indices, data)
    indices = printProgress(size, indices)

    # compute how many "tasks" to clump into each job
    groupSize = slurm.cores * slurm.sequential

    for g in group(indices, groupSize):
        l = list(g)
        print("scheduling:", path, l)

        # build the executable string
        runner = f'python {executable} {path} '
        # generate the gnu-parallel command for dispatching to many CPUs across server nodes
        parallel = Slurm.buildParallel(runner, l, {
            'ntasks': slurm.cores,
            'nodes-per-process': 1,
            'threads-per-process': 1, # <-- if you need multiple threads for a single process, change this number (NOTE: the rest of the scheduling logic does not know how to handle this yet)
        })

        # generate the bash script which will be scheduled
        script = getJobScript(parallel)

        ## uncomment for debugging the scheduler to see what bash script would have been scheduled
        # print(script)
        # exit()

        # make sure to only request the number of CPU cores necessary
        slurm.cores = min([slurm.cores, len(l)])
        Slurm.schedule(script, slurm)

        # DO NOT REMOVE. This will prevent you from overburdening the slurm scheduler. Be a good citizen.
        time.sleep(2)
