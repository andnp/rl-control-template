import sys
import json
from PyExpUtils.models.ExperimentDescription import ExperimentDescription

class ExperimentModel(ExperimentDescription):
    def __init__(self, d, path):
        super().__init__(d, path)
        self.agent = d['agent']
        self.problem = d['problem']

        self.max_steps = d.get('max_steps', 0)
        self.episodes = d.get('episodes')

def load(path=None):
    path = path if path is not None else sys.argv[1]
    with open(path, 'r') as f:
        d = json.load(f)

    exp = ExperimentModel(d, path)
    return exp