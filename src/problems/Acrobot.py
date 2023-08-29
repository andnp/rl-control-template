import numpy as np
from environments.Gym import Gym
from PyExpUtils.collection.Collector import Collector
from experiment.ExperimentModel import ExperimentModel
from problems.BaseProblem import BaseProblem

class Acrobot(BaseProblem):
    def __init__(self, exp: ExperimentModel, idx: int, collector: Collector):
        super().__init__(exp, idx, collector)
        self.env = Gym(name='Acrobot-v1', seed=self.seed, max_steps=500)
        self.actions = 3

        self.observations = (6,)
        self.gamma = 1

        ma_vel1 = 4 * np.pi
        ma_vel2 = 9 * np.pi

        self.rep_params['input_ranges'] = [
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-1, 1],
            [-ma_vel1, ma_vel1],
            [-ma_vel2, ma_vel2],
        ]
