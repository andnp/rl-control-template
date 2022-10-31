from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem
from environments.Minatar import Minatar
from PyExpUtils.utils.Collector import Collector

class Breakout(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        self.env = Minatar('breakout', self.seed)
        self.actions = self.env.env.num_actions()

        self.observations = self.env.env.state_shape()
        self.gamma = 0.99
