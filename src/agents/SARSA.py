import numpy as np
from src.utils.random import argmax, choice

class SARSA:
    def __init__(self, features, actions, params):
        self.features = features
        self.actions = actions
        self.params = params

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']

        # create initial weights
        self.w = np.zeros((actions, features))

    def selectAction(self, x):
        p = np.random.rand()
        if p < self.epsilon:
            return choice(np.arange(self.actions))

        return argmax(self.w.dot(x))

    def update(self, x, a, xp, r, gamma):
        q_a = self.w[a].dot(x)
        ap = self.selectAction(x)
        qp_ap = self.w[ap].dot(xp)

        g = r + gamma * qp_ap
        delta = g - q_a

        self.w[a] += self.alpha * delta * x
        return ap