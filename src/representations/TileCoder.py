from PyFixedReps.TileCoder import TileCoder
from PyExpUtils.utils.dict import merge

class SparseTileCoder(TileCoder):
    def __init__(self, params, rng=None):
        super().__init__(merge(params, { 'scale_output': False }), rng=rng)

    def encode(self, s, a=None):
        return super().get_indices(s, a)
