import numpy as np
from ReplayTables.PER import PrioritizedReplay, PERConfig

class PER(PrioritizedReplay):
    def __init__(self, max_size: int, lag: int, rng: np.random.Generator, config: PERConfig, collector):
        super().__init__(max_size, lag, rng, config)
        self._collector = collector