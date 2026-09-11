"""Configuration features and a censored Bayesian log-duration regression.

Normal-inverse-gamma regression is sampled using latent right-censored durations.
Only the standard library is needed; randomness is local to the estimator.
"""

import hashlib
import math
import random
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


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def _cholesky(matrix):
    lower = [[0.0] * len(matrix) for _ in matrix]
    for i, row in enumerate(matrix):
        for j in range(i + 1):
            value = row[j] - dot(lower[i][:j], lower[j][:j])
            lower[i][j] = (
                math.sqrt(max(value, 1e-12)) if i == j else value / lower[j][j]
            )
    return lower


def _forward(lower, rhs):
    result = []
    for i, value in enumerate(rhs):
        result.append((value - dot(lower[i][:i], result)) / lower[i][i])
    return result


def _backward(lower, rhs):
    result = [0.0] * len(rhs)
    for i in reversed(range(len(rhs))):
        result[i] = (
            rhs[i] - sum(lower[j][i] * result[j] for j in range(i + 1, len(rhs)))
        ) / lower[i][i]
    return result


def _solve(lower, rhs):
    return _backward(lower, _forward(lower, rhs))


def _above(rng, mean, sigma, limit):
    """Sample a normal variate above a lower bound, including extreme tails."""
    threshold = (limit - mean) / sigma
    if threshold <= 0:
        while True:
            value = rng.gauss(0, 1)
            if value >= threshold:
                return mean + sigma * value
    rate = (threshold + math.hypot(threshold, 2)) / 2
    while True:
        value = threshold + rng.expovariate(rate)
        if rng.random() <= math.exp(-0.5 * (value - rate) ** 2):
            return max(limit, mean + sigma * value)


def seconds(log_seconds):
    return math.exp(log_seconds) if log_seconds < 709 else math.inf


class DurationModel:
    def __init__(self, vectors, completed, censored):
        self.vectors = vectors
        self.observed = [
            (index, math.log(max(duration, 1e-9))) for index, duration in completed
        ]
        self.censored = [
            (index, math.log(duration)) for index, duration in censored if duration > 0
        ]
        self.indices = [i for i, _ in self.observed + self.censored]
        width = len(vectors[0])
        precision = [[0.0] * width for _ in range(width)]
        for i in range(width):
            precision[i][i] = 1e-6 if i == 0 else 1.0
        for index in self.indices:
            x = vectors[index]
            for i in range(width):
                for j in range(i + 1):
                    precision[i][j] += x[i] * x[j]
                    precision[j][i] = precision[i][j]
        self.lower = _cholesky(precision)
        initial = [y for _, y in self.observed + self.censored]
        self.mean, self.scale = self._conditional(initial)
        self.shape = 2 + len(self.indices) / 2

    def _conditional(self, ys):
        rhs = [0.0] * len(self.vectors[0])
        for index, y in zip(self.indices, ys, strict=True):
            for j, x in enumerate(self.vectors[index]):
                rhs[j] += x * y
        mean = _solve(self.lower, rhs)
        scale = max(1e-9, 1 + 0.5 * (dot(ys, ys) - dot(rhs, mean)))
        return mean, scale

    def draws(self, pending, *, count=768, components=False):
        """Joint draws preserve coefficient/scale uncertainty across experiments."""
        rng = random.Random(0)
        beta = self.mean
        sigma = math.sqrt(self.scale / max(self.shape - 1, 1))
        fixed = [y for _, y in self.observed]
        for iteration in range(count + 192):
            if self.censored:
                latent = [
                    _above(rng, dot(self.vectors[index], beta), sigma, limit)
                    for index, limit in self.censored
                ]
                mean, scale = self._conditional(fixed + latent)
            else:
                mean, scale = self.mean, self.scale
            sigma = math.sqrt(scale / rng.gammavariate(self.shape, 1))
            noise = _backward(self.lower, [rng.gauss(0, 1) for _ in mean])
            beta = [value + sigma * z for value, z in zip(mean, noise, strict=True)]
            if iteration < 192:
                continue
            durations = []
            for index, elapsed in pending:
                mu = dot(self.vectors[index], beta)
                log_time = (
                    _above(rng, mu, sigma, math.log(elapsed))
                    if elapsed > 0
                    else rng.gauss(mu, sigma)
                )
                durations.append(max(0.0, seconds(log_time) - elapsed))
            yield durations if components else sum(durations)

    def priorities(self, pending):
        """Approximate reduction in uncertainty of the sum of pending durations."""
        gradient = [0.0] * len(self.mean)
        for index in pending:
            weight = math.exp(max(-30, min(30, dot(self.vectors[index], self.mean))))
            for j, value in enumerate(self.vectors[index]):
                gradient[j] += weight * value
        scores = {}
        projected_gradient = _forward(self.lower, gradient)
        for index in pending:
            projected_x = _forward(self.lower, self.vectors[index])
            scores[index] = dot(projected_gradient, projected_x) ** 2 / (
                1 + dot(projected_x, projected_x)
            )
        return scores
