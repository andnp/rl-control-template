from functools import partial
from typing import Any, Dict, Tuple

from algorithms.BaseAgent import BaseAgent
from utils.checkpoint import checkpointable
from ReplayTables.PER import PrioritizedReplay
from PyExpUtils.utils.Collector import Collector
from ReplayTables.ReplayBuffer import ReplayBuffer
from representations.networks import NetworkBuilder

import utils.chex as cxu
from utils.jax import huber_loss, Batch

import jax
import chex
import optax
import numpy as np
import haiku as hk
import jax.numpy as jnp

@cxu.dataclass
class AgentState:
    params: Any
    target_params: Any
    optim: optax.OptState


def q_loss(q, a, r, gamma, qp):
    vp = qp.max()
    target = r + gamma * vp
    target = jax.lax.stop_gradient(target)
    delta = target - q[a]

    return huber_loss(1.0, q[a], target), {
        'delta': delta,
    }

@checkpointable(('buffer', 'steps', 'state', 'updates'))
class DQN(BaseAgent):
    def __init__(self, observations: Tuple, actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)
        self.rep_params: Dict = params['representation']
        self.optimizer_params: Dict = params['optimizer']

        self.epsilon = params['epsilon']

        # build the value function approximator
        builder = NetworkBuilder(observations, self.rep_params, seed)
        self.q = builder.addHead(lambda: hk.Linear(actions, name='q'))
        self.phi = builder.getFeatureFunction()
        net_params = builder.getParams()

        # set up the target network parameters
        self.target_refresh = params.get('target_refresh', 1)
        self.reward_clip = params.get('reward_clip', 0)

        # set up the optimizer
        self.optimizer = optax.adam(
            self.optimizer_params['alpha'],
            self.optimizer_params['beta1'],
            self.optimizer_params['beta2'],
        )
        opt_state = self.optimizer.init(net_params)

        self.state = AgentState(
            params=net_params,
            target_params=net_params,
            optim=opt_state,
        )

        # set up the experience replay buffer
        self.buffer_size = params['buffer_size']
        self.batch_size = params['batch']
        self.update_freq = params.get('update_freq', 1)

        if params.get('use_per', False):
            print('using per')
            self.buffer = PrioritizedReplay(
                max_size=self.buffer_size,
                structure=Batch,
                rng=np.random.RandomState(seed),
            )
        else:
            self.buffer = ReplayBuffer(
                max_size=self.buffer_size,
                structure=Batch,
                rng=np.random.RandomState(seed),
            )

        self.update_freq = params.get('update_freq', 1)
        self.steps = 0
        self.updates = 0

    # internal compiled version of the value function
    @partial(jax.jit, static_argnums=0)
    def _values(self, params: hk.Params, x: jax.Array):
        phi = self.phi(params, x).out
        return self.q(params, phi)

    # public facing value function approximation
    def values(self, x: np.ndarray):
        x = np.asarray(x)

        # if x is a vector, then jax handles a lack of "batch" dimension gracefully
        #   at a 5x speedup
        # if x is a tensor, jax does not handle lack of "batch" dim gracefully
        if len(x.shape) > 1:
            x = np.expand_dims(x, 0)
            q = self._values(self.state.params, x)[0]

        else:
            q = self._values(self.state.params, x)

        return jax.device_get(q)

    def _loss(self, params: hk.Params, target: hk.Params, batch: Batch, weights: jax.Array):
        phi = self.phi(params, batch.x).out
        phi_p = self.phi(target, batch.xp).out

        qs = self.q(params, phi)
        qsp = self.q(target, phi_p)

        batch_loss = jax.vmap(q_loss, in_axes=0)
        losses, metrics = batch_loss(qs, batch.a, batch.r, batch.gamma, qsp)

        chex.assert_equal_shape((weights, losses))
        loss = jnp.mean(weights * losses)

        return loss, metrics

    @partial(jax.jit, static_argnums=0)
    def _computeUpdate(self, state: AgentState, batch: Batch, weights: jax.Array):
        grad_fn = jax.grad(self._loss, has_aux=True)
        grad, metrics = grad_fn(state.params, state.target_params, batch, weights)

        updates, optim = self.optimizer.update(grad, state.optim, state.params)
        params = optax.apply_updates(state.params, updates)

        new_state = AgentState(
            params=params,
            target_params=state.target_params,
            optim=optim,
        )

        return new_state, metrics

    def update(self, x, a, xp, r, gamma):
        self.steps += 1

        # if gamma is zero, we have a terminal state
        if gamma == 0:
            xp = np.zeros_like(x)

        if self.reward_clip > 0:
            r = np.clip(r, -self.reward_clip, self.reward_clip)

        # always add to the buffer
        self.buffer.add(Batch(
            x=x,
            a=a,
            xp=xp,
            r=r,
            gamma=gamma,
        ))

        # only update every `update_freq` steps
        if self.steps % self.update_freq != 0:
            return

        # only update every `update_freq` steps
        if self.steps % self.update_freq != 0:
            return

        # skip updates if the buffer isn't full yet
        if self.buffer.size() <= self.batch_size:
            return

        self.updates += 1

        batch, idxs, weights = self.buffer.sample(self.batch_size)
        self.state, metrics = self._computeUpdate(self.state, batch, weights)

        if isinstance(self.buffer, PrioritizedReplay):
            priorities = jax.device_get(metrics['delta'])
            priorities = np.abs(priorities)
            self.buffer.update_priorities(idxs, priorities)

        if self.updates % self.target_refresh == 0:
            self.state.target_params = self.state.params
