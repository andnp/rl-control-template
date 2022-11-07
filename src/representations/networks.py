import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import chex
import jax.numpy as jnp
import haiku as hk

import utils.hk as hku

ModuleBuilder = Callable[[], Callable[[chex.Array], chex.Array]]

class NetworkBuilder:
    def __init__(self, input_shape: Tuple, params: Dict[str, Any], seed: int):
        self._input_shape = input_shape
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
        def _inner(params: Any, x: chex.Array):
            return self._feat_net.apply(params['phi'], x)

        return _inner

    def addHead(self, module: ModuleBuilder, name: Optional[str] = None):
        assert not self._retrieved_params, 'Attempted to add head after params have been retrieved'
        _state = {}

        def _builder(x: chex.Array):
            head = module()
            _state['name'] = getattr(head, 'name', None)
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

        def _inner(params: Any, x: chex.Array):
            return h_net.apply(params[name], x)

        return _inner


def reluLayers(layers: List[int], name: Optional[str] = None):
    init = hk.initializers.VarianceScaling(np.sqrt(2), 'fan_avg', 'uniform')

    out = []
    for width in layers:
        out.append(hk.Linear(width, w_init=init, b_init=init, name=name))
        out.append(jax.nn.relu)

    return out

def buildFeatureNetwork(inputs: Tuple, params: Dict[str, Any], rng: Any):
    def _inner(x: chex.Array):
        name = params['type']
        hidden = params['hidden']

        if name == 'TwoLayerRelu':
            layers = reluLayers([hidden, hidden], name='phi')

        elif name == 'OneLayerRelu':
            layers = reluLayers([hidden], name='phi')

        elif name == 'MinatarNet':
            layers = [
                hk.Conv2D(16, 3, 2, name='phi'),
                jax.nn.relu,
                hk.Flatten(name='phi'),
            ]
            layers += reluLayers([hidden], name='phi')

        else:
            raise NotImplementedError()

        return hku.accumulatingSequence(layers)(x)

    network = hk.without_apply_rng(hk.transform(_inner))

    sample_input = jnp.zeros((1,) + tuple(inputs))
    net_params = network.init(rng, sample_input)

    return network, net_params
