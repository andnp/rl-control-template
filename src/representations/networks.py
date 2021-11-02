from functools import partial
import numpy as np
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import haiku as hk

from utils.fp import pipe

def nn(layers: List[int], actions: int, x: np.ndarray):
    init = hk.initializers.VarianceScaling(np.sqrt(2), 'fan_avg', 'uniform')
    b_init = hk.initializers.Constant(0)

    hidden = []
    for layer in layers:
        hidden.append(hk.Linear(layer, w_init=init, b_init=b_init))
        hidden.append(jax.nn.relu)

    hidden = hk.Sequential(hidden)

    values = hk.Sequential([
        hk.Linear(actions, w_init=init, b_init=b_init)
    ])

    h = hidden(x)
    return values(h), h

def getNetwork(inputs: Tuple, outputs: int, params: Dict[str, Any], seed: int):
    name = params['type']

    if name == 'TwoLayerRelu':
        hidden = params['hidden']
        layers = [hidden, hidden]

        network = partial(nn, layers, outputs)

    elif name == 'OneLayerRelu':
        hidden = params['hidden']
        layers = [hidden]

        network = partial(nn, layers, outputs)

    elif name == 'MinatarNet':
        def conv(x):
            hidden = hk.Sequential([
                hk.Conv2D(16, 3, 2),
                jax.nn.relu,
                hk.Flatten(),
            ])

            return hidden(x)

        hidden = params['hidden']
        layers = [hidden]
        network = pipe([
            conv,
            partial(nn, layers, outputs)
        ])

    else:
        raise NotImplementedError()

    network = hk.without_apply_rng(hk.transform(network))
    net_params = network.init(jax.random.PRNGKey(seed), jnp.zeros((1,) + tuple(inputs)))

    return network, net_params
