from functools import partial
from typing import Any, Dict, Tuple

from ReplayTables.Table import Table
from algorithms.BaseAgent import BaseAgent
from utils.checkpoint import checkpointable
from PyExpUtils.utils.Collector import Collector
from representations.networks import NetworkBuilder
from utils.jax import Batch, vmap_except, argmax_with_random_tie_breaking


import jax
import optax
import numpy as np
import haiku as hk
import utils.hk as hku
import utils.chex as cxu

tree_leaves = jax.tree_util.tree_leaves
tree_map = jax.tree_util.tree_map

@cxu.dataclass
class AgentState:
    params: Any
    optim: optax.OptState

@checkpointable(('buffer', 'steps', 'state'))
class EQRC(BaseAgent):
    def __init__(self, observations: Tuple, actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)
        self.rep_params: Dict = params['representation']
        self.optimizer_params: Dict = params['optimizer']

        self.epsilon = params['epsilon']
        self.beta = params.get('beta', 1.)
        self.reward_clip = params.get('reward_clip', 0)

        # build the value function and h network.
        # built in three parts:
        #  (1) obs -> features (i.e. everything up to the second-to-last layer)
        #  (2) features -> q
        #  (3) features -> h
        builder = NetworkBuilder(observations, self.rep_params, seed)
        zero_init = hk.initializers.Constant(0)
        self.q = builder.addHead(lambda: hku.DuelingHeads(actions, name='q', optimistic=True))
        self.h = builder.addHead(lambda: hku.DuelingHeads(actions, name='h', w_init=zero_init, b_init=zero_init))
        self.phi = builder.getFeatureFunction()

        all_params = builder.getParams()

        # set up the optimizer
        self.stepsize = self.optimizer_params['alpha']
        self.optimizer = optax.adam(
            self.optimizer_params['alpha'],
            self.optimizer_params['beta1'],
            self.optimizer_params['beta2'],
        )
        opt_state = self.optimizer.init(all_params)

        # set up agent state
        self.state = AgentState(
            params=all_params,
            optim=opt_state,
        )

        # set up the experience replay buffer
        self.buffer_size = params['buffer_size']
        self.batch_size = params['batch']
        self.update_freq = params.get('update_freq', 1)
        self.steps = 0

        self.buffer = Table(max_size=self.buffer_size, seed=seed, columns=[
            { 'name': 'Obs', 'shape': observations },
            { 'name': 'Action', 'shape': 1, 'dtype': 'int_' },
            { 'name': 'NextObs', 'shape': observations },
            { 'name': 'Reward', 'shape': 1 },
            { 'name': 'Discount', 'shape': 1 },
        ])

    # jit'ed internal value function approximator
    # considerable speedup, especially for larger networks (note: haiku networks are not jit'ed by default)
    @partial(jax.jit, static_argnums=0)
    def _values(self, state: AgentState, x: jax.Array):
        phi = self.phi(state.params, x).out
        return self.q(state.params, phi)

    # public facing value function approximation
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

    # compute the total QRC loss for both sets of parameters (value parameters and h parameters)
    def _loss(self, params, batch: Batch):
        phi = self.phi(params, batch.x).out
        q = self.q(params, phi)
        h = self.h(params, phi)

        phi_p = self.phi(params, batch.xp).out
        qp = self.q(params, phi_p)

        # apply qc loss function to each sample in the minibatch
        # gives back value of the loss individually for parameters of v and h
        # note QC instead of QRC (i.e. no regularization)
        v_loss, h_loss, metrics = qc_loss(q, batch.a, batch.r, batch.gamma, qp, h, self.epsilon)

        h_loss = h_loss.mean()
        v_loss = v_loss.mean()

        metrics |= {
            'v_loss': v_loss,
            'h_loss': h_loss,
        }

        return v_loss + h_loss, metrics

    # compute the update and return the new parameter states
    # and optimizer state (i.e. ADAM moving averages)
    @partial(jax.jit, static_argnums=0)
    def _computeUpdate(self, state: AgentState, batch: Batch):
        params = state.params
        grad, metrics = jax.grad(self._loss, has_aux=True)(params, batch)

        updates, new_optim = self.optimizer.update(grad, state.optim, params)
        assert isinstance(updates, dict)

        decay = tree_map(
            lambda h, dh: dh - self.stepsize * self.beta * h,
            params['h'],
            updates['h'],
        )

        updates |= {'h': decay}
        new_params = optax.apply_updates(params, updates)

        new_state = AgentState(
            params=new_params,
            optim=new_optim,
        )

        return new_state, metrics

    # Public facing update function
    def update(self, x, a, xp, r, gamma):
        self.steps += 1
        # If gamma is zero, then we are at a terminal state
        # it doesn't really matter what sp is represented as, since we will multiply it by gamma=0 later anyways
        # however, setting sp = nan (which is more semantically correct) causes some issues with autograd
        if gamma == 0:
            xp = np.zeros_like(x)

        if self.reward_clip > 0:
            r = np.clip(r, -self.reward_clip, self.reward_clip)

        # always add to the buffer
        self.buffer.addTuple((x, a, xp, r, gamma))

        # only update every `update_freq` steps
        if self.steps % self.update_freq != 0:
            return

        # only update every `update_freq` steps
        if self.steps % self.update_freq != 0:
            return

        # also skip updates if the buffer isn't full yet
        if len(self.buffer) <= self.batch_size:
            return

        # draw some samples from the replay buffer and make an update
        samples = self.buffer.sample(self.batch_size)
        batch = Batch(*samples)
        self.state, metrics = self._computeUpdate(self.state, batch)
        for k, v in metrics.items():
            self.collector.collect(k, v)

# ---------------
# -- Utilities --
# ---------------

@partial(vmap_except, exclude=['epsilon'])
def qc_loss(q, a, r, gamma, qtp1, h, epsilon):
    pi = argmax_with_random_tie_breaking(qtp1)

    pi = (1.0 - epsilon) * pi + (epsilon / qtp1.shape[0])
    pi = jax.lax.stop_gradient(pi)

    vtp1 = qtp1.dot(pi)
    target = r + gamma * vtp1
    target = jax.lax.stop_gradient(target)

    delta = target - q[a]
    delta_hat = h[a]

    v_loss = 0.5 * delta**2 + gamma * jax.lax.stop_gradient(delta_hat) * vtp1
    h_loss = 0.5 * (jax.lax.stop_gradient(delta) - delta_hat)**2

    return v_loss, h_loss, {
        'delta': delta,
        'h': delta_hat,
    }
