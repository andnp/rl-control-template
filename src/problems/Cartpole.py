import numpy as np
from PyRlEnvs.domains.Cartpole import Cartpole as Env
from PyExpUtils.collection.Collector import Collector
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem

class Cartpole(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)
        self.env = Env(randomize=False, seed=self.seed)
        self.actions = 2

        x_thresh = 4.8
        theta_thresh = 12 * 2 * np.pi / 360

        # encode the observation ranges for this problem
        # useful for tile-coding
        self.rep_params['input_ranges'] = [
            [-x_thresh, x_thresh],
            [-6, 6],
            [-theta_thresh, theta_thresh],
            [-2.0, 2.0],
        ]

        self.observations = (4,)
        self.gamma = 0.999
