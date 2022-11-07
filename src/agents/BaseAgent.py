from abc import abstractmethod
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from PyExpUtils.utils.Collector import Collector
from PyFixedReps.BaseRepresentation import BaseRepresentation
from ReplayTables.LagBuffer import LagBuffer, Experience
import RlGlue.agent

from utils.policies import createEGreedy

class IdentityRep(BaseRepresentation):
    def encode(self, s, a=None):
        return s

@dataclass
class Interaction(Experience):
    s: Optional[np.ndarray]
    a: int
    r: Optional[float]
    gamma: float
    terminal: bool

class BaseAgent(RlGlue.agent.BaseAgent):
    def __init__(self, observations: Tuple, actions: int, params: Dict, collector: Collector, seed: int):
        self.observations = observations
        self.actions = actions
        self.params = params
        self.collector = collector

        self.rng = np.random.RandomState(seed)
        self.rep = IdentityRep()

        self.gamma = params.get('gamma', 1)
        self.n_step = params.get('n_step', 1)
        self.epsilon = params.get('epsilon', 0)
        self.lag = LagBuffer(lag=self.n_step)

        self._policy = createEGreedy(self.values, self.actions, self.epsilon, self.rng)

    # ----------------------
    # -- Default settings --
    # ----------------------
    def policy(self, obs: np.ndarray) -> int:
        return self._policy.selectAction(obs)

    # ------------------------
    # -- Subclass contracts --
    # ------------------------

    @abstractmethod
    def values(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def update(self, x, a, xp, r, gamma):
        pass

    @abstractmethod
    def cleanup(self):
        pass

    # ----------------------
    # -- RLGlue interface --
    # ----------------------

    def start(self, s: np.ndarray):
        self.lag.flush()

        x = self.rep.encode(s)
        x = np.asarray(x)
        a = self.policy(x)
        self.lag.add(Interaction(
            s=x,
            a=a,
            r=None,
            gamma=0,
            terminal=False,
        ))
        return a

    def step(self, r: float, sp: Optional[np.ndarray]):
        a = -1
        xp = None
        if sp is not None:
            xp = self.rep.encode(sp)
            xp = np.asarray(xp)
            a = self.policy(xp)

        interaction = Interaction(
            s=xp,
            a=a,
            r=r,
            gamma=self.gamma,
            terminal=False,
        )

        for exp in self.lag.add(interaction):
            self.update(
                x=exp.s,
                a=exp.a,
                xp=exp.sp,
                r=exp.r,
                gamma=exp.gamma,
            )

        return a

    def end(self, r: float):
        interaction = Interaction(
            s=None,
            a=-1,
            r=r,
            gamma=0,
            terminal=True,
        )
        for exp in self.lag.add(interaction):
            self.update(
                x=exp.s,
                a=exp.a,
                xp=exp.sp,
                r=exp.r,
                gamma=exp.gamma,
            )

        self.lag.flush()
