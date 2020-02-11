import numpy as np
from src.agents.registry import getAgent

class BaseProblem:
    def __init__(self, exp, idx):
        self.exp = exp
        self.idx = idx

        perm = exp.getPermutation(idx)
        self.params = perm['metaParameters']

        self.agent = None
        self.env = None
        self.rep = None
        self.gamma = None

        self.features = 0
        self.actions = 0

    def getEnvironment(self):
        return self.env

    def getRepresentation(self):
        return self.rep

    def getGamma(self):
        return self.gamma

    def getAgent(self):
        Agent = getAgent(self.exp.agent)
        self.agent = Agent(self.features, self.actions, self.params)
        return self.agent

    def getSteps(self):
        return self.exp.steps