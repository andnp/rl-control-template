from agents.SARSA import SARSA

def getAgent(name):
    if name == 'SARSA':
        return SARSA

    raise NotImplementedError()
