import sys
import os
sys.path.append(os.getcwd() + '/src')

import random
import subprocess
from functools import partial
from multiprocessing.pool import Pool

from PyExpUtils.runner import Args
from PyExpUtils.results.backends.h5 import detectMissingIndices
import experiment.ExperimentModel as Experiment

def count(pre, it):
    print(pre, 0, end='\r')
    for i, x in enumerate(it):
        print(pre, i + 1, end='\r')
        yield x

    print()

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print('Please run again using')
        print('python scripts/local.py [entry.py] [runs] [base/path/to/results] [paths/to/descriptions]...')
        exit(0)

    pool = Pool(15)

    runs = sys.argv[2]
    args = Args.ArgsModel({
        'experiment_paths': sys.argv[4:],
        'base_path': sys.argv[3],
        'runs': int(runs),
        'executable': "python " + sys.argv[1],
    })


    cmds = []
    for path in args.experiment_paths:
        exp = Experiment.load(path)

        indices = detectMissingIndices(exp, args.runs, 'step_return.h5')
        indices = count(path, indices)

        for idx in indices:
            exe = f'{args.executable} {path} {idx}'
            cmds.append(exe)

    print(len(cmds))
    random.shuffle(cmds)
    res = pool.imap_unordered(partial(subprocess.run, shell=True, stdout=subprocess.PIPE), cmds, chunksize=1)
    for i, _ in enumerate(res):
        sys.stderr.write(f'\r{i+1}/{len(cmds)}')
    sys.stderr.write('\n')
