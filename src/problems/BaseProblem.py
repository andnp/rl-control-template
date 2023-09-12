from typing import Optional
from PyExpUtils.collection.Collector import Collector
from RlGlue.environment import BaseEnvironment

from experiment.ExperimentModel import ExperimentModel
from algorithms.BaseAgent import BaseAgent
from algorithms.registry import getAgent


class BaseProblem:
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        self.exp = exp
        self.idx = idx

        self.collector = collector

        self.params = exp.get_hypers(idx)
        self.env_params = self.params.get('environment', {})
        self.exp_params = self.params.get('experiment', {})
        self.rep_params = self.params.get('representation', {})

        self.agent: Optional[BaseAgent] = None
        self.env: Optional[BaseEnvironment] = None
        self.gamma: Optional[float] = None

        self.seed = exp.getRun(idx)

        self.observations = (0,)
        self.actions = 0

    def getEnvironment(self):
        if self.env is None:
            raise Exception('Expected the environment object to be constructed already')

        return self.env

    def getAgent(self):
        if self.agent is not None:
            return self.agent

        if self.gamma is not None:
            self.params['gamma'] = self.gamma

        Agent = getAgent(self.exp.agent)
        self.agent = Agent(self.observations, self.actions, self.params, self.collector, self.seed)
        return self.agent
