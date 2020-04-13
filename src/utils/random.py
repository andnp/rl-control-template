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
    top = vals[0]
    ties = [top]
    for i, v in enumerate(vals):
        if v > top:
            top = v
            ties = [i]
        elif v == top:
            ties.append(i)

    return choice(ties)


## Functions that don't use the global numpy rng

## Functions using custom random state
# way faster than np.random.choice
# arr is an array of probabilities, should sum to 1
def sample_rng(rng, arr):
    r = rng.random()
    s = 0
    for i, p in enumerate(arr):
        s += p
        if s > r or s == 1:
            return i

    # worst case if we run into floating point error, just return the last element
    # we should never get here
    return len(arr) - 1

# also much faster than np.random.choice
# for some reason, also faster than np.random.randint(len(arr))
# choose an element from a list with uniform random probability
def uniform_choice_rng(rng, arr):
    idxs = rng.permutation(len(arr))
    return arr[idxs[0]]

def uniform_argchoice_rng(rng, arr):
    idxs = rng.permutation(len(arr))
    return idxs[0]

# argmax that breaks ties randomly
def argmax_rng(rng, vals):
    top = vals[0]
    ties = [top]
    for i, v in enumerate(vals):
        if v > top:
            top = v
            ties = [i]
        elif v == top:
            ties.append(i)

    return uniform_choice_rng(rng, ties)
