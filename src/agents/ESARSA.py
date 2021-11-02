import numpy as np
from typing import Dict, Tuple
from numba import njit
from PyExpUtils.utils.Collector import Collector

from representations.TileCoder import SparseTileCoder
from agents.BaseAgent import BaseAgent
from utils.policies import createEGreedy

# NOTE: this uses index-based features e.g. coming from a tile-coder
# would need to update this to use a standard dot-product if not
# using sparse features
@njit(cache=True)
def _update(w, x, a, xp, pi, r, gamma, alpha):
    qsa = w[a][x].sum()

    qsp = w.T[xp].sum(axis=0)

    delta = r + gamma * qsp.dot(pi) - qsa

    w[a][x] = w[a][x] + alpha / len(x) * delta

@njit(cache=True)
def value(w, x):
    return w.T[x].sum(axis=0)

class ESARSA(BaseAgent):
    def __init__(self, observations: Tuple, actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']

        # build representation
        self.rep_params: Dict = params['representation']
        self.rep = SparseTileCoder({
            'dims': observations[0],
            'tiles': self.rep_params['tiles'],
            'tilings': self.rep_params['tilings'],
            'input_ranges': self.rep_params['input_ranges'],
        })

        # create initial weights
        self.w = np.zeros((actions, self.rep.features()))

        # create a policy
        self.policy = createEGreedy(lambda state: value(self.w, state), self.actions, self.epsilon, self.rng)

    def update(self, x, a, xp, r, gamma):
        pi = self.policy.probs(xp)
        _update(self.w, x, a, xp, pi, r, gamma, self.alpha)
