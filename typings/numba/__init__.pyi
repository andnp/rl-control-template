# from typing import Callable
from typing import Callable, overload
from PyExpUtils.utils.types import T

@overload
def njit(cache: bool) -> Callable[[T], T]: ...
@overload
def njit(f: T) -> T: ...
