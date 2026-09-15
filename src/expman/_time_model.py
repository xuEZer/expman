"""Bounded configuration feature encoding for Stage estimates."""

import hashlib
import math
from collections.abc import Mapping


def _flatten(value, path=()):
    if isinstance(value, Mapping) and value:
        result = {}
        for key, item in value.items():
            result.update(_flatten(item, (*path, ("key", str(key)))))
        return result
    if isinstance(value, (list, tuple)) and value:
        result = {}
        for index, item in enumerate(value):
            result.update(_flatten(item, (*path, ("index", index))))
        return result
    return {path: value}


def _token(value):
    if type(value) is int:
        return ("int", format(value, "x"))
    if isinstance(value, (set, frozenset)):
        return (type(value).__name__, repr(sorted(_token(item) for item in value)))
    return (type(value).__name__, repr(value))


def features(configs):
    """Encode varying numeric fields and categorical values; bound matrix size."""
    flattened = [_flatten(config) for config in configs]
    paths = sorted({path for row in flattened for path in row}, key=repr)
    columns = []
    missing = object()
    for path in paths:
        values = [row.get(path, missing) for row in flattened]
        tokens = [_token(v) if v is not missing else ("missing", "") for v in values]
        if len(set(tokens)) == 1:
            continue
        if all(
            type(v) in (int, float) and (type(v) is int or math.isfinite(v))
            for v in values
        ):
            # Signed log handles zero/negative values and scale differences.
            numbers = [
                (1 if v >= 0 else -1)
                * (math.log(abs(v) + 1) if type(v) is int else math.log1p(abs(v)))
                for v in values
            ]
            low, high = min(numbers), max(numbers)
            if high > low:
                columns.append(
                    (repr(path), [(v - low) / (high - low) * 2 - 1 for v in numbers])
                )
        else:
            for token in sorted(set(tokens)):
                columns.append(
                    (repr((path, token)), [float(v == token) for v in tokens])
                )
    width = min(len(columns), 48)
    rows = [[1.0] + [0.0] * width for _ in configs]
    for index, (name, values) in enumerate(columns):
        sign = 1
        if len(columns) > width:
            digest = hashlib.sha256(name.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % width
            sign = 1 if digest[4] % 2 else -1
        for row, value in zip(rows, values, strict=True):
            row[index + 1] += sign * value
    return rows
