import sys
import os
sys.path.append(os.getcwd() + '/src')

import random
import argparse
import subprocess
from functools import partial
from multiprocessing.pool import Pool

from PyExpUtils.runner import Args
from PyExpUtils.results.backends.pandas import detectMissingIndices
import experiment.ExperimentModel as Experiment

parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, required=True)
parser.add_argument('-e', type=str, nargs='+', required=True)
parser.add_argument('--entry', type=str, default='src/main.py')
parser.add_argument('--results', type=str, default='./')

def count(pre, it):
    print(pre, 0, end='\r')
    for i, x in enumerate(it):
        print(pre, i + 1, end='\r')
        yield x

    print()

if __name__ == "__main__":
    cmdline = parser.parse_args()

    pool = Pool(6)

    args = Args.ArgsModel({
        'experiment_paths': cmdline.e,
        'base_path': cmdline.results,
        'runs': cmdline.runs,
        'executable': "python " + cmdline.entry,
    })

    cmds = []
    for path in args.experiment_paths:
        exp = Experiment.load(path)

        indices = detectMissingIndices(exp, args.runs, 'step_return')
        indices = count(path, indices)

        for idx in indices:
            exe = f'{args.executable} --silent -e {path} -i {idx}'
            cmds.append(exe)

    print(len(cmds))
    random.shuffle(cmds)
    res = pool.imap_unordered(partial(subprocess.run, shell=True, stdout=subprocess.PIPE), cmds, chunksize=1)
    for i, _ in enumerate(res):
        sys.stderr.write(f'\r{i+1}/{len(cmds)}')
    sys.stderr.write('\n')
