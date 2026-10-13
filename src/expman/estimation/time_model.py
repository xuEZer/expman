"""Bounded configuration feature encoding for Stage estimates."""

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

MAX_COLUMNS = 48


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


@dataclass(frozen=True)
class FeatureVariable:
    """One declared configuration value as it appears in an encoded row.

    Numeric values are signed-log scaled and then min-max mapped onto ``[-1, 1]``
    no matter their original scale, so one numeric value occupies one column
    spanning two units. A categorical value becomes a one-hot group: two rows
    describe the same level when they activate a column of the group in common.
    """

    kind: str
    columns: tuple[int, ...]


@dataclass(frozen=True)
class Encoding:
    """Encoded rows plus the declared variables that produced them.

    ``variables`` is ``None`` once the column count passes ``MAX_COLUMNS`` and
    groups are hashed together: membership is then no longer recoverable, and
    distances fall back to comparing columns directly.
    """

    rows: list[list[float]]
    variables: tuple[FeatureVariable, ...] | None


def encode(configs) -> Encoding:
    """Encode varying numeric fields and categorical values; bound matrix size."""
    flattened = [_flatten(config) for config in configs]
    paths = sorted({path for row in flattened for path in row}, key=repr)
    columns = []
    variables: list[FeatureVariable] = []
    missing = object()
    for path in paths:
        values: list[Any] = [row.get(path, missing) for row in flattened]
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
                # Numeric values share one span, so a variable is one column.
                variables.append(FeatureVariable("numeric", (len(columns),)))
        else:
            first = len(columns) + 1
            for token in sorted(set(tokens)):
                columns.append(
                    (repr((path, token)), [float(v == token) for v in tokens])
                )
            variables.append(
                FeatureVariable("categorical", tuple(range(first, len(columns) + 1)))
            )
    width = min(len(columns), MAX_COLUMNS)
    rows = [[1.0] + [0.0] * width for _ in configs]
    for index, (name, values) in enumerate(columns):
        sign = 1
        if len(columns) > width:
            digest = hashlib.sha256(name.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % width
            sign = 1 if digest[4] % 2 else -1
        for row, value in zip(rows, values, strict=True):
            row[index + 1] += sign * value
    return Encoding(rows, None if len(columns) > width else tuple(variables))


def variable_distance(encoding: Encoding, left, right) -> float:
    """Mean per-variable disagreement between two encoded rows, in ``[0, 1]``.

    Distance is averaged over the declared variables rather than over encoded
    columns, so a variable contributes the same weight no matter how many columns
    its encoding occupies, and each variable is bounded by one before averaging.
    Every Stage model measures configuration distance this way: an estimate is
    only as good as the neighborhood it was drawn from, and the two models must
    agree on what "nearby" means.

    Once columns are hashed together the grouping is gone and only per-column
    differences remain.
    """
    variables = encoding.variables
    if variables is None:
        spread = [abs(a - b) for a, b in zip(left[1:], right[1:], strict=True)]
        return sum(spread) / len(spread) if spread else 0.0
    if not variables:
        return 0.0
    total = 0.0
    for variable in variables:
        if variable.kind == "numeric":
            # Numeric columns were min-max mapped onto [-1, 1]: span is two.
            column = variable.columns[0]
            total += min(1.0, abs(left[column] - right[column]) / 2.0)
        elif not any(
            left[column] > 0.0 and right[column] > 0.0 for column in variable.columns
        ):
            total += 1.0
    return total / len(variables)
