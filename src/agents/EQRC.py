from functools import partial
from typing import Any, Dict
import numpy as np

from PyExpUtils.utils.Collector import Collector
from ReplayTables.Table import Table
from agents.BaseAgent import BaseAgent
from utils.policies import createEGreedy
from representations.networks import getNetwork

from utils.jax import Batch

import jax
import optax
import jax.numpy as jnp
import haiku as hk

def buildH(actions: int, x: np.ndarray):
    h = hk.Sequential([
        hk.Linear(actions, w_init=hk.initializers.Constant(0), b_init=hk.initializers.Constant(0))
    ])

    return h(jax.lax.stop_gradient(x))

def _argmax_with_random_tie_breaking(preferences):
    optimal_actions = (preferences == preferences.max(axis=-1, keepdims=True))
    return optimal_actions / optimal_actions.sum(axis=-1, keepdims=True)

def qc_loss(epsilon, q, a, r, gamma, qtp1, h):
    pi = _argmax_with_random_tie_breaking(qtp1)

    pi = (1.0 - epsilon) * pi + (epsilon / qtp1.shape[0])
    pi = jax.lax.stop_gradient(pi)

    vtp1 = qtp1.dot(pi)
    target = r + gamma * vtp1
    target = jax.lax.stop_gradient(target)

    delta = target - q[a]
    delta_hat = h[a]

    v_loss = 0.5 * delta**2 + gamma * jax.lax.stop_gradient(delta_hat) * vtp1
    h_loss = 0.5 * (jax.lax.stop_gradient(delta) - delta_hat)**2

    return v_loss, h_loss

class EQRC(BaseAgent):
    def __init__(self, observations: int, actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)
        self.rep_params: Dict = params['representation']
        self.optimizer_params: Dict = params['optimizer']

        self.epsilon = params['epsilon']
        self.beta = params.get('beta', 1.)

        # set up initialization of the value function network
        # and target network
        self.value_net, net_params = getNetwork(observations, actions, self.rep_params, seed)

        # to build the secondary weights, we need to know the size of the "feature layer" of our nn
        # there is almost certainly a better way than this, but it's fine
        _, x = self.value_net.apply(net_params, jnp.zeros(observations))

        h = partial(buildH, actions)
        self.h = hk.without_apply_rng(hk.transform(h))
        h_params = self.h.init(jax.random.PRNGKey(seed), jnp.zeros(x.shape[0]))

        self.params = {
            'w': net_params,
            'h': h_params
        }

        # set up the optimizer
        self.optimizer = optax.adam(
            self.optimizer_params['alpha'],
            self.optimizer_params['beta1'],
            self.optimizer_params['beta2'],
        )
        self.opt_state = self.optimizer.init(self.params)

        # set up the experience replay buffer
        self.buffer_size = params['buffer_size']
        self.batch_size = params['batch']

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

    # jit'ed internal value function approximator
    # considerable speedup, especially for larger networks (note: haiku networks are not jit'ed by default)
    @partial(jax.jit, static_argnums=0)
    def _values(self, params: hk.Params, x: np.ndarray):
        return self.value_net.apply(params, x)[0]

    # public facing value function approximation
    def values(self, x: np.ndarray):
        x = np.asarray(x)
        return self._values(self.params['w'], x)

    # compute the total QRC loss for both sets of parameters (value parameters and h parameters)
    def _loss(self, params, batch):
        # forward pass of value function network
        q, phi = self.value_net.apply(params['w'], batch.x)
        qtp1, _ = self.value_net.apply(params['w'], batch.xp)

        # take the "feature" layer from the value network
        # and apply a linear function approximator to obtain h(s, a)
        h = self.h.apply(params['h'], phi)

        # apply qc loss function to each sample in the minibatch
        # gives back value of the loss individually for parameters of v and h
        # note QC instead of QRC (i.e. no regularization)
        v_loss, h_loss = jax.vmap(partial(qc_loss, self.epsilon), in_axes=0)(q, batch.a, batch.r, batch.gamma, qtp1, h)

        h_loss = h_loss.mean()
        v_loss = v_loss.mean()

        regularizer = sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params['h']))

        return v_loss + h_loss + self.beta * regularizer

    # compute the update and return the new parameter states
    # and optimizer state (i.e. ADAM moving averages)
    @partial(jax.jit, static_argnums=0)
    def _computeUpdate(self, params: hk.Params, opt: Any, batch: Batch):
        delta, grad = jax.value_and_grad(self._loss)(params, batch)

        updates, state = self.optimizer.update(grad, opt, params)
        params = optax.apply_updates(params, updates)

        return jnp.sqrt(delta), state, params

    # uses fast functional API to compute new parameters
    # then assign those parameters to the stateful OOP API
    # to maintain similarity to other non-jax algorithm code
    def _updateNetwork(self, batch: Batch):
        # note that we need to pass in net_params, target_params, and opt_state as arguments here
        # we only have access to a cached version of "self" within these functions due to jax.jit
        # so we need to manually maintain the stateful portion ourselves
        delta, state, params = self._computeUpdate(self.params, self.opt_state, batch)

        self.params = params
        self.opt_state = state

        return delta

    # Public facing update function
    def update(self, s, a, sp, r, gamma):
        # If gamma is zero, then we are at a terminal state
        # it doesn't really matter what sp is represented as, since we will multiply it by gamma=0 later anyways
        # however, setting sp = nan (which is more semantically correct) causes some issues with autograd
        if gamma == 0:
            sp = np.zeros_like(s)

        # always add to the buffer
        self.buffer.addTuple((s, a, sp, r, gamma))

        # also skip updates if the buffer isn't full yet
        if len(self.buffer) > self.batch_size:
            samples = self.buffer.sample(self.batch_size)
            batch = Batch(*samples)
            self._updateNetwork(batch)
