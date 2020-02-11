from src.problems.BaseProblem import BaseProblem
from src.environments.MountainCar import MountainCar as MCEnv
from PyFixedReps.TileCoder import TileCoder

class ScaledTileCoder(TileCoder):
    def encode(self, s):
        p = s[0]
        v = s[1]

        p = (p + 1.2) / 1.7
        v = (v + 0.07) / 0.14
        return super().encode((p, v)) / float(self.num_tiling)

class MountainCar(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.env = MCEnv()
        self.actions = 3

        self.rep = ScaledTileCoder({
            'dims': 2,
            'tiles': 4,
            'tilings': 16,
        })

        self.features = self.rep.features()
        self.gamma = 1.0