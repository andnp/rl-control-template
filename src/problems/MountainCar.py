import numpy as np
from PyRlEnvs.domains.MountainCar import GymMountainCar as Env
from PyExpUtils.utils.Collector import Collector
from PyFixedReps.BaseRepresentation import BaseRepresentation
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem

def minMax(x, mi, ma):
    return (x - mi) / (ma - mi)

class MCScaledRep(BaseRepresentation):
    _u = np.array([0.5, 0.07])
    _l = np.array([-1.2, -0.07])

    def encode(self, s, a=None):
        return (s - self._l) / (self._u - self._l)

class MountainCar(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)
        self.env = Env(seed=self.seed)
        self.actions = 3

        self.rep = MCScaledRep()

        # encode the observation ranges for this problem
        # useful for tile-coding
        self.rep_params['input_ranges'] = [
            [-1, 1],
            [-1, 1],
        ]

        self.observations = (2,)
        self.gamma = 0.99
