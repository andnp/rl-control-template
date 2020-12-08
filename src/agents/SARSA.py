from typing import Dict
import numpy as np
from PyExpUtils.utils.random import argmax, choice

class SARSA:
    def __init__(self, features: int, actions: int, params: Dict, seed: int):
        self.features = features
        self.actions = actions
        self.params = params

        self.random = np.random.RandomState(seed)

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']

        # create initial weights
        self.w = np.zeros((actions, features))

    def selectAction(self, x):
        p = self.random.rand()
        if p < self.epsilon:
            return choice(np.arange(self.actions), self.random)

        return argmax(self.w.dot(x))

    def update(self, x, a, xp, r, gamma):
        q_a = self.w[a].dot(x)
        ap = self.selectAction(xp)
        qp_ap = self.w[ap].dot(xp)

        g = r + gamma * qp_ap
        delta = g - q_a

        self.w[a] += self.alpha * delta * x
        return ap
