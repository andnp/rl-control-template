import numpy as np
from src.utils.arrays import fillRest, last

class Collector:
    def __init__(self):
        self.run_data = {}
        self.all_data = {}

    def reset(self):
        for k in self.run_data:
            # if there's already an array get that
            # otherwise construct a new empty array
            arr = self.all_data.get(k, [])
            arr.append(self.run_data[k])

            # put the array back in case we were working with a new array
            self.all_data[k] = arr

        # reset the run_data for the next run
        self.run_data = {}

    def fillRest(self, value, steps):
        for k in self.run_data:
            arr = self.run_data[k]
            l = last(arr)
            v = value
            if not np.isscalar(l):
                v = np.zeros_like(l)
                v.fill(value)

            fillRest(arr, v, steps)

    def collect(self, name, value):
        arr = self.run_data.get(name, [])
        arr.append(value)

        self.run_data[name] = arr

    def getStats(self, name):
        arr = self.all_data[name]

        runs = len(arr)
        min_len = min(map(lambda a: len(a), arr))

        arr = list(map(lambda a: a[:min_len], arr))
        mean = np.mean(arr, axis=0)
        stderr = np.std(arr, axis=0, ddof=1) / np.sqrt(runs)

        return [mean, stderr, runs]
