import numpy as np

from numba import njit
from typing import Dict, Tuple
from PyExpUtils.collection.Collector import Collector

from algorithms.tc.TCAgent import TCAgent
from utils.checkpoint import checkpointable
from utils.policies import egreedy_probabilities

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

@checkpointable(('w', ))
class ESARSA(TCAgent):
    def __init__(self, observations: Tuple, actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']

        # create initial weights
        self.w = np.zeros((actions, self.rep.features()), dtype=np.float64)

    def policy(self, obs: np.ndarray) -> np.ndarray:
        qs = self.values(obs)
        return egreedy_probabilities(qs, self.actions, self.epsilon)

    def values(self, x: np.ndarray):
        x = np.asarray(x)
        return value(self.w, x)

    def update(self, x, a, xp, r, gamma):
        if xp is None:
            xp = np.zeros_like(x)
            pi = np.zeros(self.actions)
        else:
            pi = self.policy(xp)

        _update(self.w, x, a, xp, pi, r, gamma, self.alpha)
