import sys
import os
sys.path.append(os.getcwd() + '/src')

import math
import time
import argparse
import dataclasses
import PyExpUtils.runner.Slurm as Slurm
import experiment.ExperimentModel as Experiment

from functools import partial
from PyExpUtils.utils.generator import group
from PyExpUtils.runner.utils import approximate_cost, gather_missing_indices

parser = argparse.ArgumentParser()
parser.add_argument('--cluster', type=str, required=True)
parser.add_argument('--runs', type=int, required=True)
parser.add_argument('-e', type=str, nargs='+', required=True)
parser.add_argument('--entry', type=str, default='src/main.py')
parser.add_argument('--results', type=str, default='./')

cmdline = parser.parse_args()

ANNUAL_ALLOCATION = 724

# -------------------------------
# Generate scheduling bash script
# -------------------------------
cwd = os.getcwd()
project_name = os.path.basename(cwd)

venv_origin = f'{cwd}/venv.tar.xz'
venv = '$SLURM_TMPDIR'

# the contents of the string below will be the bash script that is scheduled on compute canada
# change the script accordingly (e.g. add the necessary `module load X` commands)
def getJobScript(parallel):
    return f"""#!/bin/bash

#SBATCH --signal=B:SIGUSR1@120

cd {cwd}
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 tar -xf {venv_origin} -C {venv}

export MPLBACKEND=TKAgg
export OMP_NUM_THREADS=1
{parallel}
    """

# -----------------
# Environment check
# -----------------
if not os.path.exists(venv_origin):
    print("WARNING: zipped virtual environment not found at:", venv_origin)
    print("Trying to make one now")
    code = os.system('tar -caf venv.tar.xz .venv')
    if code:
        raise Exception("Failed to make virtual env")

# ----------------
# Scheduling logic
# ----------------
slurm = Slurm.fromFile(cmdline.cluster)

# compute how many "tasks" to clump into each job
groupSize = slurm.cores * slurm.sequential

# compute how much time the jobs are going to take
hours, minutes, seconds = slurm.time.split(':')
total_hours = int(hours) + (int(minutes) / 60) + (int(seconds) / 3600)

# gather missing
missing = gather_missing_indices(cmdline.e, cmdline.runs, result_file='steps', loader=Experiment.load)

# compute cost
memory = Slurm.memory_in_mb(slurm.mem_per_core)
compute_cost = partial(approximate_cost, cores_per_job=slurm.cores, mem_per_core=memory, hours=total_hours)
cost = sum(compute_cost(math.ceil(len(job_list) / groupSize)) for job_list in missing.values())
perc = (cost / ANNUAL_ALLOCATION) * 100

print(f"Expected to use {cost:.2f} core years, which is {perc:.4f}% of our annual allocation")
input("Press Enter to confirm or ctrl+c to exit")

# start scheduling
for path in missing:
    for g in group(missing[path], groupSize):
        l = list(g)
        print("scheduling:", path, l)
        # make sure to only request the number of CPU cores necessary
        cores = min([slurm.cores, len(l)])
        sub = dataclasses.replace(slurm, cores=cores)

        # build the executable string
        # instead of activating the venv every time, just use its python directly
        runner = f'{venv}/.venv/bin/python {cmdline.entry} -e {path} --save_path {cmdline.results} --checkpoint_path=$SCRATCH/checkpoints/{project_name} -i '

        # generate the gnu-parallel command for dispatching to many CPUs across server nodes
        parallel = Slurm.buildParallel(runner, l, sub)

        # generate the bash script which will be scheduled
        script = getJobScript(parallel)

        # uncomment for debugging the scheduler to see what bash script would have been scheduled
        # print(Slurm.to_cmdline_flags(sub))
        # print(script)
        # exit()

        Slurm.schedule(script, sub)

        # DO NOT REMOVE. This will prevent you from overburdening the slurm scheduler. Be a good citizen.
        time.sleep(2)
