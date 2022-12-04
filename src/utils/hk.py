from typing import Callable, Dict, Optional, Sequence
import jax.numpy as jnp
import haiku as hk
import chex
from utils.chex import dataclass

Init = hk.initializers.Initializer
Layer = Callable[[chex.Array], chex.Array]


@dataclass
class AccumulatedOutput:
    activations: Dict[str, chex.Array]
    out: chex.Array

def accumulatingSequence(fs: Sequence[Layer]):
    def _inner(x: chex.Array):
        out: Dict[str, chex.Array] = {}

        y = x
        for f in fs:
            y = f(y)
            if isinstance(f, hk.Module):
                out[f.name] = y

        return AccumulatedOutput(activations=out, out=y)
    return _inner


class DuelingHeads(hk.Module):
    def __init__(
        self,
        output_size: int,
        w_init: Optional[Init] = None,
        b_init: Optional[Init] = None,
        name: Optional[str] = None,
        optimistic: bool = False
    ):
        super().__init__(name=name)

        self.input_size = None
        self.output_size = output_size
        self.w_init = w_init or hk.initializers.VarianceScaling()
        self.b_init = b_init or hk.initializers.VarianceScaling()
        if optimistic:
            assert b_init is None, 'Cannot specify optimism and a custom bias initialization'
            self.b_init = hk.initializers.Constant(1)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        if not inputs.shape:
            raise ValueError('Input must not be scalar.')

        self.input_size = inputs.shape[-1]
        dtype = inputs.dtype

        wa = hk.get_parameter('wa', [self.input_size, self.output_size], dtype, init=self.w_init)
        ba = hk.get_parameter('ba', [self.output_size], dtype, init=self.b_init)

        wv = hk.get_parameter('wv', [self.input_size, 1], dtype, init=self.w_init)
        bv = hk.get_parameter('bv', [1], dtype, init=self.b_init)

        adv = inputs.dot(wa) + ba
        v = inputs.dot(wv) + bv

        return v + (adv - jnp.mean(adv, axis=-1, keepdims=True))
