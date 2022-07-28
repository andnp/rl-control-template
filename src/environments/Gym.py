from typing import Optional
import gym
from RlGlue.environment import BaseEnvironment

class Gym(BaseEnvironment):
    def __init__(self, name: str, seed: int, max_steps: Optional[int] = None):
        self.env = gym.envs.make(name)
        self.seed = seed

        self.max_steps = max_steps
        if max_steps is not None:
            self.env._max_episode_steps = max_steps

    def start(self):
        return self.env.reset(seed=self.seed)

    def step(self, a):
        sp, r, t, info = self.env.step(a)

        return (r, sp, t)
