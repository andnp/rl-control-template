from typing import Type
from algorithms.BaseAgent import BaseAgent

from algorithms.nn.DQN import DQN
from algorithms.nn.EQRC import EQRC

from algorithms.tc.ESARSA import ESARSA
from algorithms.tc.SoftmaxAC import SoftmaxAC

def getAgent(name) -> Type[BaseAgent]:
    if name == 'DQN':
        return DQN

    if name == 'EQRC':
        return EQRC

    if name == 'ESARSA':
        return ESARSA

    if name == 'SoftmaxAC':
        return SoftmaxAC

    raise Exception('Unknown algorithm')
