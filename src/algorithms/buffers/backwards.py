import numpy as np
from ReplayTables.BackwardsReplay import BackwardsReplay, BackwardsReplayConfig
from ReplayTables.ReplayBuffer import Batch


class BackwardsER(BackwardsReplay):
    def __init__(self, max_size: int, lag: int, rng: np.random.Generator, config: BackwardsReplayConfig, collector):
        super().__init__(max_size, lag, rng, config)
        self._collector = collector

    def update_priorities(self, batch: Batch, priorities: np.ndarray):
        ...
