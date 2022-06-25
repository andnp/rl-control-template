from functools import partial
from typing import Any, Dict, Tuple
import numpy as np
import copy

from PyExpUtils.utils.Collector import Collector
from ReplayTables.Table import Table
from agents.BaseAgent import BaseAgent
from representations.networks import getNetwork
from utils.policies import createEGreedy

from utils.jax import huber_loss, Batch

import jax
import optax
import jax.numpy as jnp
import haiku as hk

def q_loss(q, a, r, gamma, qp):
    vp = qp.max()
    target = r + gamma * vp
    target = jax.lax.stop_gradient(target)

    return huber_loss(1.0, q[a], target)

class DQN(BaseAgent):
    def __init__(self, observations: Tuple, actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)
        self.rep_params: Dict = params['representation']
        self.optimizer_params: Dict = params['optimizer']

        self.epsilon = params['epsilon']

        # set up initialization of the value function network
        # and target network
        self.value_net, self.net_params = getNetwork(observations, actions, self.rep_params, seed)
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
        self.update_freq = params.get('update_freq', 1)

        # an empty tuple is treated as a null dimension. So these end up as
        # a vector of length buffer_size, instead of a (buffer_size x 1) matrix
        self.buffer = Table(max_size=self.buffer_size, seed=seed, columns=[
            { 'name': 'Obs', 'shape': observations },
            { 'name': 'Action', 'shape': 1, 'dtype': 'int_' },
            { 'name': 'NextObs', 'shape': observations },
            { 'name': 'Reward', 'shape': 1 },
            { 'name': 'Discount', 'shape': 1 },
        ])

        # build the policy
        self.policy = createEGreedy(self.values, self.actions, self.epsilon, self.rng)

        self.steps = 0

    # internal compiled version of the value function
    @partial(jax.jit, static_argnums=0)
    def _values(self, params: hk.Params, x: np.ndarray):
        return self.value_net.apply(params, x)[0]

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

    def _loss(self, params: hk.Params, target: hk.Params, batch: Batch):
        qs, _ = self.value_net.apply(params, batch.x)
        qsp, _ = self.value_net.apply(target, batch.xp)

        losses = jax.vmap(q_loss, in_axes=0)(qs, batch.a, batch.r, batch.gamma, qsp)

        return losses.mean()

    @partial(jax.jit, static_argnums=0)
    def _computeUpdate(self, params: hk.Params, target: hk.Params, opt: Any, batch: Batch):
        delta, grad = jax.value_and_grad(self._loss)(params, target, batch)

        updates, state = self.optimizer.update(grad, opt, params)
        params = optax.apply_updates(params, updates)

        return jnp.sqrt(delta), state, params

    def updateNetwork(self, batch: Batch):
        # note that we need to pass in net_params, target_params, and opt_state as arguments here
        # we only have access to a cached version of "self" within these functions due to jax.jit
        # so we need to manually maintain the stateful portion ourselves
        delta, state, params = self._computeUpdate(self.net_params, self.target_params, self.opt_state, batch)

        self.net_params = params
        self.opt_state = state

        if self.steps % self.target_refresh == 0:
            self.target_params = params

        return delta

    def update(self, s, a, sp, r, gamma):
        self.steps += 1

        # if gamma is zero, we have a terminal state
        if gamma == 0:
            sp = np.zeros_like(s)

        # always add to the buffer
        self.buffer.addTuple((s, a, sp, r, gamma))

        # only update every `update_freq` steps
        if self.steps % self.update_freq != 0:
            return

        # skip updates if the buffer isn't full yet
        if len(self.buffer) > self.batch_size:
            samples = self.buffer.sample(self.batch_size)
            batch = Batch(*samples)
            self.updateNetwork(batch)
