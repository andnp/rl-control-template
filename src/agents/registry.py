from agents.DQN import DQN
from agents.EQRC import EQRC
from agents.ESARSA import ESARSA

def getAgent(name):
    if name == 'ESARSA':
        return ESARSA

    if name == 'EQRC':
        return EQRC

    if name == 'DQN':
        return DQN

    raise NotImplementedError()
