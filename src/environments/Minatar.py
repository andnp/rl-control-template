from rlglue import BaseEnvironment
from minatar import Environment

class Minatar(BaseEnvironment):
    def __init__(self, name, seed):
        self.env = Environment(name, random_seed=seed)

    def start(self):
        self.env.reset()
        s = self.env.state()
        return s.astype('float32')

    def step(self, action):
        r, t = self.env.act(action)
        sp = self.env.state().astype('float32')

        return (sp, r, t, False, {})
