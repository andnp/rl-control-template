from src.problems.BaseProblem import BaseProblem
from src.environments.MountainCar import MountainCar as MCEnv
from PyFixedReps.TileCoder import TileCoder

class MountainCar(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.env = MCEnv(self.seed)
        self.actions = 3

        self.rep = TileCoder({
            'dims': 2,
            'tiles': 4,
            'tilings': 16,
            'input_ranges': [(-1.2, 0.5), (-0.07, 0.07)],
            'scale_output': True,
        })

        self.features = self.rep.features()
        self.gamma = 1.0
