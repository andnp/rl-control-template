import gym
from RlGlue import BaseEnvironment

class Gym(BaseEnvironment):
    def __init__(self, name, seed, max_steps=None):
        self.env = gym.envs.make(name)
        self.env.seed(seed)

        self.max_steps = max_steps
        if max_steps is not None:
            self.env._max_episode_steps = max_steps

    def start(self):
        return self.env.reset()

    def step(self, a):
        sp, r, t, info = self.env.step(a)

        return (r, sp, t)
