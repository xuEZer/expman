"""Read-only, pickle-compatible views of configuration values."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, eq=False)
class FrozenDict(Mapping):
    _data: Mapping

    def __init__(self, values, *, tracker=None, path=()):
        object.__setattr__(self, "_tracker", tracker)
        object.__setattr__(self, "_path", path)
        if any(not isinstance(key, str) for key in values):
            raise TypeError("configuration keys must be strings")
        object.__setattr__(
            self,
            "_data",
            MappingProxyType(
                {
                    key: freeze(value, tracker=tracker, path=(*path, key))
                    for key, value in values.items()
                }
            ),
        )

    def _read(self, path=None):
        if self._tracker is not None:
            self._tracker.record(self._path if path is None else path)

    def __getitem__(self, key):
        self._read((*self._path, key))
        return self._data[key]

    def get(self, *args, **kwargs):
        raise TypeError(
            "ctx.cfg.get() is not allowed; declare required keys and use cfg[key]"
        )

    def __contains__(self, key):
        raise TypeError(
            "membership checks on ctx.cfg are not allowed; use explicit configuration values"
        )

    def keys(self):
        self._read()
        return super().keys()

    def items(self):
        self._read()
        return super().items()

    def values(self):
        self._read()
        return super().values()

    def __repr__(self):
        self._read()
        return f"FrozenDict({self._data!r})"

    def __iter__(self):
        self._read()
        return iter(self._data)

    def __len__(self):
        self._read()
        return len(self._data)

    def __reduce__(self):
        self._read()
        return FrozenDict, (dict(self._data),)

    def __deepcopy__(self, memo):
        self._read()
        return self


@dataclass(frozen=True, eq=False)
class FrozenList(Sequence):
    _values: tuple

    def __init__(self, values, *, tracker=None, path=()):
        object.__setattr__(self, "_tracker", tracker)
        object.__setattr__(self, "_path", path)
        object.__setattr__(
            self,
            "_values",
            tuple(
                freeze(value, tracker=tracker, path=(*path, index))
                for index, value in enumerate(values)
            ),
        )

    def _read(self, path=None):
        if self._tracker is not None:
            self._tracker.record(self._path if path is None else path)

    def __repr__(self):
        self._read()
        return f"FrozenList({self._values!r})"

    def __getitem__(self, key):
        self._read() if isinstance(key, slice) else self._read((*self._path, key))
        value = self._values[key]
        return FrozenList(value) if isinstance(key, slice) else value

    def __len__(self):
        self._read()
        return len(self._values)

    def __eq__(self, other):
        self._read()
        return isinstance(other, (list, tuple, FrozenList)) and self._values == tuple(
            other
        )

    def __reduce__(self):
        self._read()
        return FrozenList, (self._values,)

    def __deepcopy__(self, memo):
        self._read()
        return self


def freeze(value: Any, *, tracker=None, path=()) -> Any:
    if isinstance(value, (FrozenDict, FrozenList)):
        return value
    if isinstance(value, Mapping):
        return FrozenDict(value, tracker=tracker, path=path)
    if isinstance(value, (list, tuple)):
        return FrozenList(value, tracker=tracker, path=path)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze(item) for item in value)
    if value is None or isinstance(
        value, (str, bytes, bool, int, float, date, datetime)
    ):
        return value
    raise TypeError(
        f"mutable configuration object {type(value).__name__}; put runtime objects in ctx.state"
    )
