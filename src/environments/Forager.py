import numpy as np
from typing import Any
from RlGlue import BaseEnvironment
from forager.Env import ForagerEnv
from forager.config import ForagerConfig
from forager.objects import Wall, Flower, Thorns

class Forager(BaseEnvironment):
    def __init__(self, seed: int):
        config = ForagerConfig(
            size=500,
            object_types={
                'wall': Wall,
                'flower': Flower,
                'thorns': Thorns,
            },

            observation_mode='objects',
            aperture=9,
            seed=seed,
        )

        self.env = ForagerEnv(config)
        self.env.generate_objects(name='flower', freq=0.1)
        self.env.generate_objects(name='thorns', freq=0.2)
        self.env.generate_objects(name='wall', freq=0.01)

    def start(self) -> Any:
        obs = self.env.start()
        return obs.astype(np.float32)

    def step(self, a: int):
        obs, r = self.env.step(a)
        return (r, obs.astype(np.float32), False, {})
