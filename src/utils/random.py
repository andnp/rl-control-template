import numpy as np

# way faster than np.random.choice
# arr is an array of probabilities, should sum to 1
def sample(arr):
    r = np.random.random()
    s = 0
    for i, p in enumerate(arr):
        s += p
        if s > r or s == 1:
            return i

    # worst case if we run into floating point error, just return the last element
    # we should never get here
    return len(arr) - 1

# also much faster than np.random.choice
# choose an element from a list with uniform random probability
def choice(arr):
    idxs = np.random.permutation(len(arr))
    return arr[idxs[0]]

# argmax that breaks ties randomly
def argmax(vals):
    ties = []
    top = vals[0]
    for i, v in enumerate(vals):
        if v > top:
            top = v
            ties = [i]
        elif v == top:
            ties.append(i)

    return choice(ties)