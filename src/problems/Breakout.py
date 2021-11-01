from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem
from environments.Minatar import Minatar
from PyFixedReps.BaseRepresentation import BaseRepresentation
from PyExpUtils.utils.Collector import Collector

class IdentityRep(BaseRepresentation):
    def encode(self, s, a=None):
        return s

class Breakout(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        self.env = Minatar('breakout', self.seed)
        self.actions = self.env.env.num_actions()

        self.rep = IdentityRep()

        self.observations = self.env.env.state_shape()
        self.gamma = 0.99
