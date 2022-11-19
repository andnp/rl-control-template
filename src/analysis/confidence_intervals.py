import numpy as np
import numba

@numba.njit(cache=True)
def bootstrapCI(data: np.ndarray, bootstraps: int = 10000, seed: int = 0):
    np.random.seed(seed)
    assert len(data.shape) == 2
    seeds, measurements = data.shape

    out = np.empty((3, measurements))
    for i in range(measurements):
        bs = np.empty(bootstraps)
        for j in range(bootstraps):
            sub = np.random.choice(data[:, i], size=seeds, replace=True)
            bs[j] = np.mean(sub)

        out[0, i] = np.percentile(bs, 2.5)
        out[1, i] = np.mean(data[:, i])
        out[2, i] = np.percentile(bs, 97.5)

    return out
