from typing import Optional
from PyExpUtils.utils.Collector import Collector
from PyFixedReps.BaseRepresentation import BaseRepresentation
from RlGlue.environment import BaseEnvironment

from experiment.ExperimentModel import ExperimentModel
from agents.BaseAgent import BaseAgent
from agents.registry import getAgent


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

        self.agent: Optional[BaseAgent] = None
        self.env: Optional[BaseEnvironment] = None
        self.gamma: Optional[float] = None
        self.rep = IdentityRep()

        self.seed = exp.getRun(idx)

        self.observations = 0
        self.actions = 0

    def getEnvironment(self):
        if self.env is None:
            raise Exception('Expected the environment object to be constructed already')

        return self.env

    def getRepresentation(self):
        return self.rep

    def getAgent(self):
        Agent = getAgent(self.exp.agent)
        self.agent = Agent(self.observations, self.actions, self.params, self.collector, self.seed)
        return self.agent
