from abc import abstractmethod
import numpy as np
from typing import Dict, Tuple
from PyExpUtils.utils.Collector import Collector
from PyFixedReps.BaseRepresentation import BaseRepresentation

class IdentityRep(BaseRepresentation):
    def encode(self, s, a=None):
        return s

class BaseAgent:
    def __init__(self, observations: Tuple, actions: int, params: Dict, collector: Collector, seed: int):
        self.observations = observations
        self.actions = actions
        self.params = params
        self.collector = collector

        self.rng = np.random.RandomState(seed)
        self.rep = IdentityRep()

    @abstractmethod
    def update(self, x, a, xp, r, gamma):
        pass

    @abstractmethod
    def cleanup(self):
        pass
