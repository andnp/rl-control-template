from .iid import IIDBuffer, IIDConfig
from .per import PER, PERConfig
from .backwards import BackwardsER, BackwardsReplayConfig
from .pser import PSER, PSERConfig

def getBufferBuilder(name: str):
    buffers = {
        'iid': (IIDBuffer, IIDConfig),
        'per': (PER, PERConfig),
        'backwards': (BackwardsER, BackwardsReplayConfig),
        'pser': (PSER, PSERConfig)
    }

    return buffers[name]