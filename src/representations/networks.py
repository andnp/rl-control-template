import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import haiku as hk

import utils.hk as hku

ModuleBuilder = Callable[[], Callable[[jax.Array | np.ndarray], jax.Array]]

class NetworkBuilder:
    def __init__(self, input_shape: Tuple, params: Dict[str, Any], seed: int):
        self._input_shape = tuple(input_shape)
        self._h_params = params
        self._rng, feat_rng = jax.random.split(jax.random.PRNGKey(seed))

        self._feat_net, self._feat_params = buildFeatureNetwork(input_shape, params, feat_rng)

        self._params = {
            'phi': self._feat_params,
        }

        self._retrieved_params = False

    def getParams(self):
        self._retrieved_params = True
        return self._params

    def getFeatureFunction(self):
        def _inner(params: Any, x: jax.Array | np.ndarray):
            return self._feat_net.apply(params['phi'], x)

        return _inner

    def addHead(self, module: ModuleBuilder, name: Optional[str] = None, grad: bool = True):
        assert not self._retrieved_params, 'Attempted to add head after params have been retrieved'
        _state = {}

        def _builder(x: jax.Array | np.ndarray):
            head = module()
            _state['name'] = getattr(head, 'name', None)

            if not grad:
                x = jax.lax.stop_gradient(x)

            out = head(x)
            return out

        sample_in = jnp.zeros((1,) + self._input_shape)
        sample_phi = self._feat_net.apply(self._feat_params, sample_in).out

        self._rng, rng = jax.random.split(self._rng)
        h_net = hk.without_apply_rng(hk.transform(_builder))
        h_params = h_net.init(rng, sample_phi)

        name = name or _state.get('name')
        assert name is not None, 'Could not detect name from module'
        self._params[name] = h_params

        def _inner(params: Any, x: jax.Array):
            return h_net.apply(params[name], x)

        return _inner


def reluLayers(layers: List[int], name: Optional[str] = None):
    w_init = hk.initializers.Orthogonal(np.sqrt(2))
    b_init = hk.initializers.Constant(0)

    out = []
    for width in layers:
        out.append(hk.Linear(width, w_init=w_init, b_init=b_init, name=name))
        out.append(jax.nn.relu)

    return out

def buildFeatureNetwork(inputs: Tuple, params: Dict[str, Any], rng: Any):
    def _inner(x: jax.Array):
        name = params['type']
        hidden = params['hidden']

        if name == 'TwoLayerRelu':
            layers = reluLayers([hidden, hidden], name='phi')

        elif name == 'OneLayerRelu':
            layers = reluLayers([hidden], name='phi')

        elif name == 'MinatarNet':
            w_init = hk.initializers.Orthogonal(np.sqrt(2))
            layers = [
                hk.Conv2D(16, 3, 2, w_init=w_init, name='phi'),
                jax.nn.relu,
                hk.Flatten(name='phi'),
            ]
            layers += reluLayers([hidden], name='phi')

        elif name == 'ForagerNet':
            w_init = hk.initializers.Orthogonal(np.sqrt(2))
            layers = [
                hk.Conv2D(16, 3, 2, w_init=w_init, name='phi'),
                jax.nn.relu,
                hk.Flatten(name='phi'),
            ]
            layers += reluLayers([hidden], name='phi')

        elif name == 'AtariNet':
            w_init = hk.initializers.Orthogonal(np.sqrt(2))
            layers = [
                lambda x: x.astype(np.float32),
                make_conv(32, (8, 8), (4, 4)),
                jax.nn.relu,
                make_conv(64, (4, 4), (2, 2)),
                jax.nn.relu,
                make_conv(64, (3, 3), (1, 1)),
                jax.nn.relu,
                hk.Flatten(),
                hk.Linear(512, w_init=w_init),
                jax.nn.relu,
            ]

        else:
            raise NotImplementedError()

        return hku.accumulatingSequence(layers)(x)

    network = hk.without_apply_rng(hk.transform(_inner))

    sample_input = jnp.zeros((1,) + tuple(inputs))
    net_params = network.init(rng, sample_input)

    return network, net_params


def make_conv(size: int, shape: Tuple[int, int], stride: Tuple[int, int]):
    w_init = hk.initializers.Orthogonal(np.sqrt(2))
    b_init = hk.initializers.Constant(0)
    return hk.Conv2D(
        size,
        kernel_shape=shape,
        stride=stride,
        w_init=w_init,
        b_init=b_init,
        padding='VALID',
        name='conv',
    )
