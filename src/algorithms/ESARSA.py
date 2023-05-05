import numpy as np

from numba import njit
from typing import Dict, Tuple
from algorithms.BaseAgent import BaseAgent
from utils.checkpoint import checkpointable
from PyExpUtils.utils.Collector import Collector
from PyFixedReps.TileCoder import TileCoderConfig
from representations.TileCoder import SparseTileCoder

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
class ESARSA(BaseAgent):
    def __init__(self, observations: Tuple, actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)

        # define parameter contract
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']

        # build representation
        self.rep_params: Dict = params['representation']
        self.rep = SparseTileCoder(TileCoderConfig(
            tiles=self.rep_params['tiles'],
            tilings=self.rep_params['tilings'],
            dims=observations[0],
            input_ranges=self.rep_params['input_ranges']
        ))

        # create initial weights
        self.w = np.zeros((actions, self.rep.features()), dtype=np.float64)

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
