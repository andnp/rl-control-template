import numpy as np
from dataclasses import dataclass
from ReplayTables.ReplayBuffer import ReplayBuffer, Batch

@dataclass
class IIDConfig:
    ...

class IIDBuffer(ReplayBuffer):
    def __init__(self, max_size: int, lag: int, rng: np.random.Generator, config: IIDConfig, collector):
        super().__init__(max_size, lag, rng)
        self._collector = collector

    def update_priorities(self, batch: Batch, priorities: np.ndarray):
        ...