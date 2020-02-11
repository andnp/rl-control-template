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

slurm_path = sys.argv[1]
executable = sys.argv[2]
base_path = sys.argv[3]
runs = sys.argv[4]
experiment_paths = sys.argv[5:]

# TODO: change name of results file
def generateMissing(paths):
    raise Exception('Set the name of the results file saved from these experiments')
    for i, p in enumerate(paths):
        summary_path = p + '/result.npy'
        if not os.path.exists(summary_path):
            yield i

def printProgress(size, it):
    for i, _ in enumerate(it):
        print(f'{i + 1}/{size}', end='\r')
        if i - 1 == size:
            print()
        yield _

cwd = os.getcwd()
def getJobScript(parallel):
    return f"""#!/bin/bash
cd {cwd}
{parallel}
    """

for path in experiment_paths:
    print(path)
    exp = Experiment.load(path)
    slurm = Slurm.fromFile(slurm_path)

    size = exp.numPermutations() * runs

    paths = listResultsPaths(exp, runs)
    paths = printProgress(size, paths)
    indices = generateMissing(paths)

    groupSize = slurm.tasks * slurm.tasksPerNode

    for g in group(indices, groupSize):
        l = list(g)
        print("scheduling:", path, l)
        parallel = Slurm.buildParallel(executable, l, {
            'ntasks': slurm.tasks,
            'nodes-per-process': 1,
            'threads-per-process': 1,
        })
        script = getJobScript(parallel)
        slurm.tasks = min([slurm.tasks, len(l)])
        Slurm.schedule(script, slurm)
        time.sleep(2)
