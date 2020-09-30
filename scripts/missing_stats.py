import sys
import os
sys.path.append(os.getcwd())

import src.experiment.ExperimentModel as Experiment
from PyExpUtils.results.paths import listResultsPaths

if len(sys.argv) < 3:
    print('Please run again using')
    print('python scripts/missing_stats.py [base_path] [paths/to/descriptions]...')
    exit(0)

base_path = sys.argv[1]

# TODO: change name of results file
def generateMissing(paths):
    raise Exception('Set the name of the results file saved from these experiments')
    for i, p in enumerate(paths):
        summary_path = p + '/result.npy'
        if not os.path.exists(summary_path):
            yield i

def count(gen):
    c = 0
    for _ in gen:
        c += 1

    return c

experiment_paths = sys.argv[2:]

total = 0
total_missing = 0
for path in experiment_paths:
    print(path)
    exp = Experiment.load(path)

    size = exp.numPermutations()

    paths = listResultsPaths(exp, 1)
    indices = generateMissing(paths)
    missing = count(indices)

    total_missing += missing
    total += size

    print(missing, size, missing / size)

print('total:', total_missing, total)
