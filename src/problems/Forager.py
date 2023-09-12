from PyExpUtils.collection.Collector import Collector
from environments.Forager import Forager as Env
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem

class Forager(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)

        self.env = Env(self.seed)
        self.actions = 4

        self.observations = (9, 9, 4)
        self.gamma = 0.99
