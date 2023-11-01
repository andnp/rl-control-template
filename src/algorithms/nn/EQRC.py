from functools import partial
from typing import Dict, Tuple
from PyExpUtils.collection.Collector import Collector
from ReplayTables.interface import Batch

from algorithms.nn.NNAgent import NNAgent, AgentState
from representations.networks import NetworkBuilder
from utils.jax import vmap_except, argmax_with_random_tie_breaking

import jax
import optax
import numpy as np
import haiku as hk
import utils.hk as hku

tree_leaves = jax.tree_util.tree_leaves
tree_map = jax.tree_util.tree_map

class EQRC(NNAgent):
    def __init__(self, observations: Tuple, actions: int, params: Dict, collector: Collector, seed: int):
        super().__init__(observations, actions, params, collector, seed)
        self.beta = params.get('beta', 1.)
        self.stepsize = self.optimizer_params['alpha']

    # ------------------------
    # -- NN agent interface --
    # ------------------------
    def _build_heads(self, builder: NetworkBuilder) -> None:
        zero_init = hk.initializers.Constant(0)
        self.q = builder.addHead(lambda: hku.DuelingHeads(self.actions, name='q', w_init=zero_init, b_init=zero_init))
        self.h = builder.addHead(lambda: hku.DuelingHeads(self.actions, name='h', w_init=zero_init, b_init=zero_init), grad=False)

    # jit'ed internal value function approximator
    # considerable speedup, especially for larger networks (note: haiku networks are not jit'ed by default)
    @partial(jax.jit, static_argnums=0)
    def _values(self, state: AgentState, x: jax.Array):
        phi = self.phi(state.params, x).out
        return self.q(state.params, phi)

    def update(self):
        self.steps += 1

        # only update every `update_freq` steps
        if self.steps % self.update_freq != 0:
            return

        # skip updates if the buffer isn't full yet
        if self.buffer.size() <= self.batch_size:
            return

        batch = self.buffer.sample(self.batch_size)
        self.state, metrics = self._computeUpdate(self.state, batch)

        metrics = jax.device_get(metrics)

        priorities = metrics['delta']
        self.buffer.update_batch(batch, priorities=priorities)

        for k, v in metrics.items():
            self.collector.collect(k, np.mean(v).item())

    # -------------
    # -- Updates --
    # -------------

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
