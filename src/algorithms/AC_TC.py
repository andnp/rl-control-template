import numpy as np

from numba import njit
from typing import Dict, Tuple
from utils.checkpoint import checkpointable
from PyExpUtils.utils.Collector import Collector
from PyFixedReps.TileCoder import TileCoderConfig

from algorithms.BaseAgent import BaseAgent
from representations.TileCoder import SparseTileCoder

@njit(cache=True)
def _update(w, theta, x, a, n_a, xp, r, pi, gamma, alpha):
    v = w[x].sum()
    vp = w[xp].sum()

    delta = r + gamma * vp - v
    w[x] += alpha * delta

    for ap in range(n_a):
        if a == ap:
            theta[ap][x] += alpha * delta * (1 - pi[ap])
        else:
            theta[ap][x] += alpha * delta * (0 - pi[ap])

    return delta

@njit(cache=True)
def compute_logits(theta, x):
    return theta.T[x].sum(axis=0)

@njit(cache=True)
def softmax(logits: np.ndarray, tau: float):
    c = logits.max()
    num = np.exp((logits - c) / tau)
    den = num.sum()
    return num / den

@checkpointable(('w', 'theta'))
class AC_TC(BaseAgent):
    def __init__(self, observations: Tuple, actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)

        # define parameter contract
        self.alpha = params['alpha']
        self.tau = params['tau']

        # build representation
        self.rep_params: Dict = params['representation']
        self.rep = SparseTileCoder(TileCoderConfig(
            tiles=self.rep_params['tiles'],
            tilings=self.rep_params['tilings'],
            dims=observations[0],
            input_ranges=self.rep_params['input_ranges']
        ))

        # create initial weights
        self.w = np.zeros((self.rep.features()), dtype=np.float32)
        self.theta = np.zeros((actions, self.rep.features()), dtype=np.float32)

    def policy(self, x: np.ndarray):
        l = compute_logits(self.theta, x)
        return softmax(l, self.tau)

    def update(self, x, a, xp, r, gamma):
        if xp is None:
            xp = np.zeros_like(x)
            pi = np.zeros(self.actions)
        else:
            pi = self.policy(xp)

        _update(self.w, self.theta, x, a, self.actions, xp, r, pi, gamma, self.alpha)
