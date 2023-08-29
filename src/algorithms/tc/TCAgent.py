import numpy as np

from abc import abstractmethod
from typing import Any, Dict, Tuple
from PyExpUtils.collection.Collector import Collector
from PyExpUtils.utils.random import sample
from ReplayTables.interface import Timestep
from ReplayTables.ingress.LagBuffer import LagBuffer

from algorithms.BaseAgent import BaseAgent
from representations.TileCoder import SparseTileCoder, TileCoderConfig
from utils.checkpoint import checkpointable

@checkpointable(('rep', 'lag'))
class TCAgent(BaseAgent):
    def __init__(self, observations: Tuple[int, ...], actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)
        self.lag = LagBuffer(self.n_step)

        self.rep_params: Dict = params['representation']
        self.rep = SparseTileCoder(TileCoderConfig(
            tiles=self.rep_params['tiles'],
            tilings=self.rep_params['tilings'],
            dims=observations[0],
            input_ranges=self.rep_params['input_ranges'],
        ))

    @abstractmethod
    def policy(self, obs: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def update(self, x, a, xp, r, gamma):
        ...

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(self, s: np.ndarray):
        self.lag.flush()

        x = self.rep.encode(s)
        pi = self.policy(x)
        a = sample(pi, rng=self.rng)
        self.lag.add(Timestep(
            x=x,
            a=a,
            r=None,
            gamma=0,
            terminal=False,
        ))
        return a

    def step(self, r: float, sp: np.ndarray | None, extra: Dict[str, Any]):
        a = -1

        # sample next action
        xp = None
        if sp is not None:
            xp = self.rep.encode(sp)
            pi = self.policy(xp)
            a = sample(pi, rng=self.rng)

        # see if the problem specified a discount term
        gamma = extra.get('gamma', 1.0)

        interaction = Timestep(
            x=xp,
            a=a,
            r=r,
            gamma=self.gamma * gamma,
            terminal=False,
        )

        for exp in self.lag.add(interaction):
            self.update(
                x=exp.x,
                a=exp.a,
                xp=exp.n_x,
                r=exp.r,
                gamma=exp.gamma,
            )

        return a

    def end(self, r: float, extra: Dict[str, Any]):
        interaction = Timestep(
            x=None,
            a=-1,
            r=r,
            gamma=0,
            terminal=True,
        )
        for exp in self.lag.add(interaction):
            self.update(
                x=exp.x,
                a=exp.a,
                xp=exp.n_x,
                r=exp.r,
                gamma=exp.gamma,
            )

        self.lag.flush()
