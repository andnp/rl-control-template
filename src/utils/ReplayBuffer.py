import numpy as np
from typing import Dict, Generic, List, TypeVar

T = TypeVar('T')

# much much faster than np.random.choice
def choice(arr: Dict[int, T], size: int = 1, rng=np.random) -> List[T]:
    idxs = rng.permutation(len(arr))
    return [arr[i] for i in idxs[:size]]

class ReplayBuffer(Generic[T]):
    def __init__(self, buffer_size: int, seed: int):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer: Dict[int, T] = {}

        self.random = np.random.RandomState(seed)

    def __len__(self):
        return len(self.buffer)

    def add(self, args: T):
        self.buffer[self.location] = args
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size: int):
        return choice(self.buffer, batch_size, self.random), []

    # match api with prioritized ER buffer
    def update_priorities(self, idxes, priorities):
        pass
