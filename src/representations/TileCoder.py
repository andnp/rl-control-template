from PyFixedReps.TileCoder import TileCoder, TileCoderConfig

class SparseTileCoder(TileCoder):
    def __init__(self, params: TileCoderConfig, rng=None):
        params.scale_output = False
        super().__init__(params, rng=rng)

    def encode(self, s):
        return super().get_indices(s)
