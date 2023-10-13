import numpy as np
from ReplayTables.PSER import PrioritizedSequenceReplay, PSERConfig

class PSER(PrioritizedSequenceReplay):
    def __init__(self, max_size: int, lag: int, rng: np.random.Generator, config: PSERConfig, collector):
        super().__init__(max_size, lag, rng, config)
        self._collector = collector