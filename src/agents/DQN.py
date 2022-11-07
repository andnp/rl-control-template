from functools import partial
from typing import Any, Dict, Tuple
import numpy as np
import copy

from PyExpUtils.utils.Collector import Collector
from ReplayTables.ReplayBuffer import ReplayBuffer
from ReplayTables.PER import PrioritizedReplay
from agents.BaseAgent import BaseAgent
from representations.networks import NetworkBuilder

from utils.jax import huber_loss, Batch

import jax
import chex
import optax
import jax.numpy as jnp
import haiku as hk


def q_loss(q, a, r, gamma, qp):
    vp = qp.max()
    target = r + gamma * vp
    target = jax.lax.stop_gradient(target)
    delta = target - q[a]

    return huber_loss(1.0, q[a], target), {
        'delta': delta,
    }

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
        self.net_params = builder.getParams()

        # set up the target network parameters
        self.target_params = copy.deepcopy(self.net_params)
        self.target_refresh = params.get('target_refresh', 1)

        # set up the optimizer
        self.optimizer = optax.adam(
            self.optimizer_params['alpha'],
            self.optimizer_params['beta1'],
            self.optimizer_params['beta2'],
        )
        self.opt_state = self.optimizer.init(self.net_params)

        # set up the experience replay buffer
        self.buffer_size = params['buffer_size']
        self.batch_size = params['batch']

        if params.get('use_per', False):
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

    # internal compiled version of the value function
    @partial(jax.jit, static_argnums=0)
    def _values(self, params: hk.Params, x: chex.Array):
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
            return self._values(self.net_params, x)[0]

        return self._values(self.net_params, x)

    def _loss(self, params: hk.Params, target: hk.Params, batch: Batch, weights: chex.Array):
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
    def _computeUpdate(self, params: hk.Params, target: hk.Params, opt: Any, batch: Batch, weights: chex.Array):
        grad, metrics = jax.grad(self._loss, has_aux=True)(params, target, batch, weights)

        updates, state = self.optimizer.update(grad, opt, params)
        params = optax.apply_updates(params, updates)

        return state, params, metrics

    def updateNetwork(self, batch: Batch, weights: np.ndarray):
        # note that we need to pass in net_params, target_params, and opt_state as arguments here
        # we only have access to a cached version of "self" within these functions due to jax.jit
        # so we need to manually maintain the stateful portion ourselves
        state, params, metrics = self._computeUpdate(self.net_params, self.target_params, self.opt_state, batch, weights)

        self.net_params = params
        self.opt_state = state

        if self.steps % self.target_refresh == 0:
            self.target_params = params

        return metrics

    def update(self, x, a, xp, r, gamma):
        self.steps += 1

        # if gamma is zero, we have a terminal state
        if gamma == 0:
            xp = np.zeros_like(x)

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

        # skip updates if the buffer isn't full yet
        if self.buffer.size() > self.batch_size:
            batch, idxs, weights = self.buffer.sample(self.batch_size)
            metrics = self.updateNetwork(batch, weights)

            if isinstance(self.buffer, PrioritizedReplay):
                priorities = jax.device_get(metrics['delta'])
                priorities = np.abs(priorities)
                self.buffer.update_priorities(idxs, priorities)
