from inspect import signature, Parameter
from typing import Callable, List, NamedTuple, Sequence, TypeVar, Union
import numpy as np

import jax
import jax.numpy as jnp

Batch = NamedTuple('Batch', [
    ('x', np.ndarray),
    ('a', np.ndarray),
    ('xp', np.ndarray),
    ('r', np.ndarray),
    ('gamma', np.ndarray),
])

def mse_loss(pred: np.ndarray, target: np.ndarray):
    return 0.5 * jnp.mean(jnp.square(pred - target))

def huber_loss(tau: float, pred: np.ndarray, target: np.ndarray):
    diffs = jnp.abs(pred - target)

    quadratic = jnp.minimum(diffs, tau)
    linear = diffs - quadratic

    losses = 0.5 * quadratic**2 + tau * linear

    return jnp.mean(losses)

def takeAlongAxis(a: np.ndarray, ind: np.ndarray):
    return jnp.squeeze(jnp.take_along_axis(a, ind[..., None], axis=-1), axis=-1)


F = TypeVar('F', bound=Callable)
def vmap_except(f: F, exclude: Sequence[str]) -> F:
    sig = signature(f)
    args = [
        k for k, p in sig.parameters.items() if p.default is Parameter.empty
    ]

    total: List[Union[int, None]] = [0] * len(args)
    for i, k in enumerate(args):
        if k in exclude:
            total[i] = None

    return jax.vmap(f, in_axes=total)
