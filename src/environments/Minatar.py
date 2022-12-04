from RlGlue import BaseEnvironment
from minatar import Environment

class Minatar(BaseEnvironment):
    def __init__(self, name, seed):
        self.env = Environment(name, random_seed=seed)

    def start(self):
        self.env.reset()
        s = self.env.state()
        return s.astype('float32')

    def step(self, a):
        r, t = self.env.act(a)
        sp = self.env.state().astype('float32')

        return (r, sp, t, {})
