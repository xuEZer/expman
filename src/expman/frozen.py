"""Read-only, pickle-compatible views of configuration values."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, eq=False)
class FrozenDict(Mapping):
    _data: Mapping

    def __init__(self, values):
        if any(not isinstance(key, str) for key in values):
            raise TypeError("configuration keys must be strings")
        object.__setattr__(
            self,
            "_data",
            MappingProxyType({key: freeze(value) for key, value in values.items()}),
        )

    def __getitem__(self, key):
        return self._data[key]

    def get(self, *args, **kwargs):
        raise TypeError(
            "ctx.cfg.get() is not allowed; declare required keys and use cfg[key]"
        )

    def __contains__(self, key):
        raise TypeError(
            "membership checks on ctx.cfg are not allowed; use explicit configuration values"
        )

    def __repr__(self):
        return f"FrozenDict({self._data!r})"

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __reduce__(self):
        return FrozenDict, (dict(self._data),)

    def __deepcopy__(self, memo):
        return self


@dataclass(frozen=True, eq=False)
class FrozenList(Sequence):
    _values: tuple

    def __init__(self, values):
        object.__setattr__(
            self,
            "_values",
            tuple(freeze(value) for value in values),
        )

    def __repr__(self):
        return f"FrozenList({self._values!r})"

    def __getitem__(self, key):
        value = self._values[key]
        return FrozenList(value) if isinstance(key, slice) else value

    def __len__(self):
        return len(self._values)

    def __eq__(self, other):
        return isinstance(other, (list, tuple, FrozenList)) and self._values == tuple(
            other
        )

    def __reduce__(self):
        return FrozenList, (self._values,)

    def __deepcopy__(self, memo):
        return self


def freeze(value: Any) -> Any:
    if isinstance(value, (FrozenDict, FrozenList)):
        return value
    if isinstance(value, Mapping):
        return FrozenDict(value)
    if isinstance(value, (list, tuple)):
        return FrozenList(value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze(item) for item in value)
    if value is None or isinstance(
        value, (str, bytes, bool, int, float, date, datetime)
    ):
        return value
    raise TypeError(
        f"mutable configuration object {type(value).__name__}; put runtime objects in ctx.state"
    )
