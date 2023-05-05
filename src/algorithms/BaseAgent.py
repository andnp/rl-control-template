from abc import abstractmethod
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from PyExpUtils.utils.Collector import Collector
from PyFixedReps.BaseRepresentation import BaseRepresentation
from ReplayTables.LagBuffer import LagBuffer, Experience
import RlGlue.agent

from utils.policies import egreedy_probabilities, sample

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
    def __init__(self, observations: Tuple[int, ...], actions: int, params: Dict, collector: Collector, seed: int):
        self.observations = observations
        self.actions = actions
        self.params = params
        self.collector = collector

        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.rep = IdentityRep()

        self.gamma = params.get('gamma', 1)
        self.n_step = params.get('n_step', 1)
        self.epsilon = params.get('epsilon', 0)
        self.reward_clip = params.get('reward_clip', 0)
        self.lag = LagBuffer(lag=self.n_step)

    # ----------------------
    # -- Default settings --
    # ----------------------
    def policy(self, obs: np.ndarray) -> np.ndarray:
        q = self.values(obs)
        pi = egreedy_probabilities(q, self.actions, self.epsilon)
        return pi

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
        pi = self.policy(x)
        a = sample(pi, rng=self.rng)
        self.lag.add(Interaction(
            s=x,
            a=a,
            r=None,
            gamma=0,
            terminal=False,
        ))
        return a

    def step(self, r: float, sp: Optional[np.ndarray], extra: Dict[str, Any]):
        a = -1

        # sample next action
        xp = None
        if sp is not None:
            xp = self.rep.encode(sp)
            xp = np.asarray(xp)
            pi = self.policy(xp)
            a = sample(pi, rng=self.rng)

        # see if the problem specified a discount term
        gamma = extra.get('gamma', 1.0)

        # possibly process the reward
        if self.reward_clip > 0:
            r = np.clip(r, -self.reward_clip, self.reward_clip)

        interaction = Interaction(
            s=xp,
            a=a,
            r=r,
            gamma=self.gamma * gamma,
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

    def end(self, r: float, extra: Dict[str, Any]):
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

    # -------------------
    # -- Checkpointing --
    # -------------------

    def __getstate__(self):
        return {
            '__args': (self.observations, self.actions, self.params, self.collector, self.seed),
            'rng': self.rng,
            'rep': self.rep,
            'lag': self.lag,
        }

    def __setstate__(self, state):
        self.__init__(*state['__args'])
        self.rng = state['rng']
        self.rep = state['rep']
        self.lag = state['lag']
