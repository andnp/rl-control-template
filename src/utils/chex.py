from typing import Any, Callable, Type, TypeVar
import chex
import dataclasses
import typing_extensions

T = TypeVar('T')

@typing_extensions.dataclass_transform(
    eq_default=True,
    order_default=False,
    field_specifiers=(dataclasses.Field, dataclasses.field),
)
def dataclass(cls: Any) -> Callable[[Type[T]], Type[T]]:
    return chex.dataclass(cls)
