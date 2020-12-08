import sys
import os
sys.path.append(os.getcwd() + '/src')

import subprocess
from functools import partial
from multiprocessing.pool import Pool

from PyExpUtils.runner import Args
from PyExpUtils.results.indices import listIndices
from PyExpUtils.results.paths import listResultsPaths
from PyExpUtils.utils.arrays import first
from PyExpUtils.utils.csv import buildCsvParams
import experiment.ExperimentModel as Experiment

def generateMissing(exp, indices, data):
    for idx in indices:
        csv_params = buildCsvParams(exp, idx)
        run = exp.getRun(idx)
        key = f'{csv_params},{run}'
        if not any(s.startswith(key) for s in data):
            yield idx

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

    pool = Pool()

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

        paths = listResultsPaths(exp, args.runs)
        res_path = first(paths)

        data = []

        raise NotImplementedError('Make sure to change the expected result file!!')
        data_path = f'{res_path}/TODO-CHANGE-ME.csv'
        if os.path.exists(data_path):
            f = open(data_path, 'r')
            data = f.readlines()
            f.close()

        indices = listIndices(exp, args.runs)
        # get all of the indices corresponding to missing results
        indices = generateMissing(exp, indices, data)
        indices = count(path, indices)

        for idx in indices:
            exe = f'{args.executable} {path} {idx}'
            cmds.append(exe)

    print(len(cmds))
    res = pool.imap_unordered(partial(subprocess.run, shell=True, stdout=subprocess.PIPE), cmds)
    for i, _ in enumerate(res):
        sys.stderr.write(f'\r{i+1}/{len(cmds)}')
    sys.stderr.write('\n')
