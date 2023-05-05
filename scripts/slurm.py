import time
import sys
import os
sys.path.append(os.getcwd() + '/src')

import math
import argparse
import numpy as np
import experiment.ExperimentModel as Experiment
import PyExpUtils.runner.Slurm as Slurm
from PyExpUtils.results.backends.pandas import detectMissingIndices
from PyExpUtils.utils.generator import group

parser = argparse.ArgumentParser()
parser.add_argument('--cluster', type=str, required=True)
parser.add_argument('--runs', type=int, required=True)
parser.add_argument('-e', type=str, nargs='+', required=True)
parser.add_argument('--entry', type=str, default='src/main.py')
parser.add_argument('--results', type=str, default='./')

cmdline = parser.parse_args()

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

# prints a progress bar
def printProgress(size, it):
    for i, _ in enumerate(it):
        print(f'{i + 1}/{size}', end='\r')
        if i - 1 == size:
            print()
        yield _

def estimateUsage(indices, groupSize, cores, hours):
    jobs = math.ceil(len(indices) / groupSize)

    total_cores = jobs * cores
    core_hours = total_cores * hours

    core_years = core_hours / (24 * 365)
    allocation = 724

    return core_years, 100 * core_years / allocation

def gatherMissing(experiment_paths, runs, groupSize, cores, total_hours):
    out = {}

    approximate_cost = np.zeros(2)

    for path in experiment_paths:
        exp = Experiment.load(path)

        indices = detectMissingIndices(exp, runs, 'steps')
        indices = sorted(indices)
        out[path] = indices

        approximate_cost += estimateUsage(indices, groupSize, cores, total_hours)

        # figure out how many indices to expect
        size = exp.numPermutations() * runs

        # log how many are missing
        print(path, f'{len(indices)} / {size}')

    return out, approximate_cost

# ----------------
# Scheduling logic
# ----------------
slurm = Slurm.fromFile(cmdline.cluster)

# compute how many "tasks" to clump into each job
groupSize = slurm.cores * slurm.sequential

# compute how much time the jobs are going to take
hours, minutes, seconds = slurm.time.split(':')
total_hours = int(hours) + (int(minutes) / 60) + (int(seconds) / 3600)

# gather missing and sum up cost
missing, cost = gatherMissing(cmdline.e, cmdline.runs, groupSize, slurm.cores, total_hours)

print(f"Expected to use {cost[0]:.2f} core years, which is {cost[1]:.4f}% of our annual allocation")
input("Press Enter to confirm or ctrl+c to exit")

for path in missing:
    # reload this because we do bad mutable things later on
    slurm = Slurm.fromFile(cmdline.cluster)

    for g in group(missing[path], groupSize):
        l = list(g)
        print("scheduling:", path, l)

        # build the executable string
        runner = f'python {cmdline.entry} -e {path} --save_path {cmdline.results} --checkpoint_path=$SCRATCH/checkpoints/experience-ordering -i '
        # generate the gnu-parallel command for dispatching to many CPUs across server nodes
        parallel = Slurm.buildParallel(runner, l, {
            'ntasks': slurm.cores,
            'nodes-per-process': 1,
            'threads-per-process': 1,
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
