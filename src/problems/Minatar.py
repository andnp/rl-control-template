from functools import partial
from environments.Minatar import Minatar
from problems.BaseProblem import BaseProblem
from PyExpUtils.collection.Collector import Collector
from experiment.ExperimentModel import ExperimentModel

class MinatarProblem(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector, game: str):
        super().__init__(exp, idx, collector)

        self.env = Minatar(game, self.seed)
        self.actions = self.env.env.num_actions()

        self.observations = self.env.env.state_shape()
        self.gamma = 0.99


Breakout = partial(MinatarProblem, game='breakout')
Seaquest = partial(MinatarProblem, game='seaquest')
Asterix = partial(MinatarProblem, game='asterix')
Freeway = partial(MinatarProblem, game='freeway')
SpaceInvaders = partial(MinatarProblem, game='space_invaders')
