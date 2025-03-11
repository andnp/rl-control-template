from typing import Optional
import gymnasium
from rlglue.environment import BaseEnvironment

class Gym(BaseEnvironment):
    def __init__(self, name: str, seed: int, max_steps: Optional[int] = None):
        self.env = gymnasium.make(name, max_episode_steps=max_steps)
        self.seed = seed

        self.max_steps = max_steps

    def start(self):
        self.seed += 1
        s, info = self.env.reset(seed=self.seed)
        return s

    def step(self, action):
        sp, r, t, trunc, info = self.env.step(action)
        return sp, float(r), t, trunc, info
