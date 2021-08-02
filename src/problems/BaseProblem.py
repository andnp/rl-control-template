from PyExpUtils.utils.Collector import Collector
from experiment.ExperimentModel import ExperimentModel
from agents.registry import getAgent

from PyFixedReps.BaseRepresentation import BaseRepresentation

class IdentityRep(BaseRepresentation):
    def encode(self, s, a=None):
        return s

class BaseProblem:
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        self.exp = exp
        self.idx = idx

        self.collector = collector

        perm = exp.getPermutation(idx)
        self.params = perm['metaParameters']
        self.env_params = self.params.get('environment', {})
        self.exp_params = self.params.get('experiment', {})
        self.rep_params = self.params.get('representation', {})

        self.agent = None
        self.env = None
        self.gamma = None
        self.rep = IdentityRep()

        self.seed = exp.getRun(idx)

        self.features = 0
        self.actions = 0

    def getEnvironment(self):
        return self.env

    def getRepresentation(self):
        return self.rep

    def getAgent(self):
        Agent = getAgent(self.exp.agent)
        self.agent = Agent(self.features, self.actions, self.params, self.collector, self.seed)
        return self.agent
