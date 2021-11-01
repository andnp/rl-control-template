from typing import Any, Callable, Sequence

F = Callable[[Any], Any]
def pipe(fs: Sequence[F]) -> F:
    def sub(x):
        out = x
        for f in fs:
            out = f(out)

        return out

    return sub
