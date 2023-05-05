from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem
from environments.Minatar import Minatar
from PyExpUtils.utils.Collector import Collector

def build_minatar_problem(game: str):
    class MinatarProblem(BaseProblem):
        def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
            super().__init__(exp, idx, collector)

            self.env = Minatar(game, self.seed)
            self.actions = self.env.env.num_actions()

            self.observations = self.env.env.state_shape()
            self.gamma = 0.99

    return MinatarProblem

Breakout = build_minatar_problem('breakout')
Seaquest = build_minatar_problem('seaquest')
Asterix = build_minatar_problem('asterix')
Freeway = build_minatar_problem('freeway')
SpaceInvaders = build_minatar_problem('space_invaders')
