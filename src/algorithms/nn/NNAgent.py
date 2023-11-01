import jax
import optax
import numpy as np
import utils.chex as cxu

from abc import abstractmethod
from typing import Any, Dict, Tuple
from PyExpUtils.collection.Collector import Collector
from ReplayTables.interface import Timestep
from ReplayTables.registry import build_buffer

from algorithms.BaseAgent import BaseAgent
from representations.networks import NetworkBuilder
from utils.checkpoint import checkpointable
from utils.policies import egreedy_probabilities, sample

@cxu.dataclass
class AgentState:
    params: Any
    optim: optax.OptState


@checkpointable(('buffer', 'steps', 'state', 'updates'))
class NNAgent(BaseAgent):
    def __init__(self, observations: Tuple[int, ...], actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)

        # ------------------------------
        # -- Configuration Parameters --
        # ------------------------------
        self.rep_params: Dict = params['representation']
        self.optimizer_params: Dict = params['optimizer']

        self.epsilon = params['epsilon']
        self.reward_clip = params.get('reward_clip', 0)

        # ---------------------
        # -- NN Architecture --
        # ---------------------
        builder = NetworkBuilder(observations, self.rep_params, seed)
        self._build_heads(builder)
        self.phi = builder.getFeatureFunction()
        net_params = builder.getParams()

        # ---------------
        # -- Optimizer --
        # ---------------
        self.optimizer = optax.adam(
            self.optimizer_params['alpha'],
            self.optimizer_params['beta1'],
            self.optimizer_params['beta2'],
        )
        opt_state = self.optimizer.init(net_params)

        # ------------------
        # -- Data ingress --
        # ------------------
        self.buffer_size = params['buffer_size']
        self.batch_size = params['batch']
        self.update_freq = params.get('update_freq', 1)

        self.buffer = build_buffer(
            buffer_type=params['buffer_type'],
            max_size=self.buffer_size,
            lag=self.n_step,
            rng=self.rng,
            config=params.get('buffer_config', {}),
        )

        # --------------------------
        # -- Stateful information --
        # --------------------------
        self.state = AgentState(
            params=net_params,
            optim=opt_state,
        )

        self.steps = 0
        self.updates = 0

    # ------------------------
    # -- NN agent interface --
    # ------------------------

    @abstractmethod
    def _build_heads(self, builder: NetworkBuilder) -> None:
        ...

    @abstractmethod
    def _values(self, state: Any, x: np.ndarray) -> jax.Array:
        ...

    @abstractmethod
    def update(self) -> None:
        ...

    def policy(self, obs: np.ndarray) -> np.ndarray:
        q = self.values(obs)
        pi = egreedy_probabilities(q, self.actions, self.epsilon)
        return pi

    # --------------------------
    # -- Base agent interface --
    # --------------------------
    def values(self, x: np.ndarray):
        x = np.asarray(x)

        # if x is a vector, then jax handles a lack of "batch" dimension gracefully
        #   at a 5x speedup
        # if x is a tensor, jax does not handle lack of "batch" dim gracefully
        if len(x.shape) > 1:
            x = np.expand_dims(x, 0)
            q = self._values(self.state, x)[0]

        else:
            q = self._values(self.state, x)

        return jax.device_get(q)

    # ----------------------
    # -- RLGlue interface --
    # ----------------------
    def start(self, x: np.ndarray):
        self.buffer.flush()
        x = np.asarray(x)
        pi = self.policy(x)
        a = sample(pi, rng=self.rng)
        self.buffer.add_step(Timestep(
            x=x,
            a=a,
            r=None,
            gamma=self.gamma,
            terminal=False,
        ))
        return a

    def step(self, r: float, xp: np.ndarray | None, extra: Dict[str, Any]):
        a = -1

        # sample next action
        if xp is not None:
            xp = np.asarray(xp)
            pi = self.policy(xp)
            a = sample(pi, rng=self.rng)

        # see if the problem specified a discount term
        gamma = extra.get('gamma', 1.0)

        # possibly process the reward
        if self.reward_clip > 0:
            r = np.clip(r, -self.reward_clip, self.reward_clip)

        self.buffer.add_step(Timestep(
            x=xp,
            a=a,
            r=r,
            gamma=self.gamma * gamma,
            terminal=False,
        ))

        self.update()
        return a

    def end(self, r: float, extra: Dict[str, Any]):
        # possibly process the reward
        if self.reward_clip > 0:
            r = np.clip(r, -self.reward_clip, self.reward_clip)

        self.buffer.add_step(Timestep(
            x=np.zeros(self.observations),
            a=-1,
            r=r,
            gamma=0,
            terminal=True,
        ))

        self.update()
