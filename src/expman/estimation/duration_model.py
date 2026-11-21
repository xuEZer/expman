"""Kernel-weighted Stage duration estimates with conformal intervals.

Duration responds multiplicatively to configuration changes: a wider batch does
not add a fixed number of seconds, it multiplies them. The model therefore works
in log space, where residuals are additive and a distance-weighted geometric mean
is the natural center.

Choosing a fixed number of nearest configurations makes the estimate jump
whenever a sample crosses the neighborhood boundary, so every completed sample
contributes instead through a Gaussian kernel. Its bandwidth is the distance of
the ``max(2, ceil(n / neighborhood))``-th positive distance, capped at the
number of distinct-from-query samples. Exact matches retain full weight.

The interval is multiplicative: the residual distribution of the center estimate
is read at the empirical order statistic single-sided conformal prediction asks
for, and applied to the prediction, so a Stage that takes hours gets an interval
in hours rather than one inherited from a Stage that takes seconds. Residuals are
leave-one-out and measured at each sample's *own* configuration, because the
error a sample carries is its own; measuring it at the query would fold the
configuration distance into the width. Completed attempts set the center and upper residual quantile. A cancelled
attempt contributes a lower bound to the residual pool for the lower endpoint.
"""

import math
from dataclasses import dataclass
from sys import float_info
from typing import NamedTuple

from expman.estimation.time_model import Encoding, variable_distance
from expman.settings import settings

# Fit the running-time curve at the 10th and 90th percentiles for every query.
RUNNING_FIT_COVERAGE = 0.8


@dataclass(frozen=True)
class StageDuration:
    """One Stage's predicted duration together with the interval around it."""

    center: float
    lower: float
    upper: float

    def remaining(self, elapsed: float, coverage: float) -> tuple[float, float]:
        """Quantiles of remaining time, conditional on still running at elapsed.

        The input interval supplies the 10th and 90th percentiles around the
        predicted median. Fit a log-logistic curve on each side. The requested
        coverage selects quantiles on that same curve. A side without measured
        spread uses a factor of two at its reference percentile. Cap each
        log-scale at 1/2 to keep the fitted distribution's mean finite.
        """
        tail = (1 - coverage) / 2
        log_odds = math.log1p(RUNNING_FIT_COVERAGE) - math.log1p(-RUNNING_FIT_COVERAGE)
        log_center = math.log(self.center)

        def scale(log_width: float) -> float:
            if log_width <= 0:
                log_width = math.log(2)
            return min(0.5, log_width / log_odds)

        left = scale(log_center - math.log(self.lower)) if self.lower > 0 else 0.5
        right = scale(math.log(self.upper) - log_center)

        def quantile(probability: float, log_survival: float) -> float:
            # If the odds of finishing before elapsed are O, the p-quantile
            # given survival has target odds (O + p) / (1 - p). The equivalent
            # forms below avoid overflowing O or subtracting nearly equal times.
            if elapsed >= self.center:
                inverse_odds = (self.center / elapsed) ** (1 / right)
                increment = math.log1p(probability * inverse_odds) - log_survival
                remaining = elapsed * math.expm1(right * increment)
            else:
                fraction = elapsed / self.center
                odds = fraction ** (1 / left)
                target_log_odds = math.log(odds + probability) - log_survival
                if target_log_odds >= 0:
                    # The quantile crosses the median. Add the gaps on either side.
                    remaining = self.center * (
                        (1 - fraction) + math.expm1(right * target_log_odds)
                    )
                elif odds > probability:
                    increment = math.log1p(probability / odds) - log_survival
                    remaining = elapsed * math.expm1(left * increment)
                else:
                    remaining = self.center * (
                        math.exp(left * target_log_odds) - fraction
                    )
            return min(remaining, float_info.max)

        return quantile(tail, math.log1p(-tail)), quantile(1 - tail, math.log(tail))


class _Sample(NamedTuple):
    row: int
    seconds: float
    log_seconds: float


def _order_statistic(values, probability: float) -> float:
    """The ``⌈p·(n+1)⌉``-th smallest value, clamped to the observed range.

    This is the single-sided conformal position. Clamping matters for small
    samples, where the requested side mass is not yet representable and the most
    extreme observation is the best available statement.
    """
    ordered = sorted(values)
    position = math.ceil(probability * (len(ordered) + 1)) - 1
    return ordered[max(0, min(len(ordered) - 1, position))]


class DurationModel:
    """Per-Stage duration model over the Stage's own measured samples.

    ``encoding`` supplies one row per run and ``indices`` maps a run id to its
    row. Samples are added with :meth:`record`; a cancelled attempt is only a
    lower bound on its own duration and is recorded as censored.
    """

    def __init__(self, encoding: Encoding, indices: dict, *, coverage: float) -> None:
        self.encoding = encoding
        self.indices = indices
        self.coverage = coverage
        self._completed: list[_Sample] = []
        self._censored: list[_Sample] = []
        self._calibrated: list[float] | None = None

    def record(self, run_id, seconds, *, censored: bool = False) -> None:
        """Add one measured duration; a censored one is only a lower bound."""
        row = self.indices.get(run_id)
        if row is None or isinstance(seconds, bool):
            return
        if not isinstance(seconds, (int, float)):
            return
        value = float(seconds)
        if not math.isfinite(value) or value <= 0:
            return
        sample = _Sample(row, value, math.log(value))
        (self._censored if censored else self._completed).append(sample)
        self._calibrated = None

    def _weights(self, query: int, samples) -> list[float]:
        """Gaussian kernel weights over every sample, keyed to one query row."""
        rows = self.encoding.rows
        distances = [
            variable_distance(self.encoding, rows[query], rows[sample.row])
            for sample in samples
        ]
        positive = sorted(distance for distance in distances if distance > 0.0)
        if not positive:
            return [1.0] * len(distances)
        rank = max(2, math.ceil(len(samples) / settings.neighborhood))
        bandwidth = positive[min(rank, len(positive)) - 1]
        return [math.exp(-0.5 * (distance / bandwidth) ** 2) for distance in distances]

    @staticmethod
    def _center(weights, samples, anchor):
        """Kernel-weighted geometric mean in log space."""
        total = sum(weights)
        if total <= 0.0:
            return None, 0.0
        numerator = sum(
            weight * (sample.log_seconds - anchor.log_seconds)
            for weight, sample in zip(weights, samples, strict=True)
        )
        offset = numerator / total
        return anchor.seconds * math.exp(offset), offset

    def _residuals(self) -> list[float]:
        """Leave-one-out log residuals, each measured at its own configuration."""
        if self._calibrated is None:
            samples = self._completed
            calibrated = []
            for index, sample in enumerate(samples):
                others = [
                    other for position, other in enumerate(samples) if position != index
                ]
                if not others:
                    continue
                prediction, _offset = self._center(
                    self._weights(sample.row, others), others, others[0]
                )
                if prediction is not None:
                    calibrated.append(sample.log_seconds - math.log(prediction))
            self._calibrated = calibrated
        return self._calibrated

    def _censored_residuals(self) -> list[float]:
        """Lower bounds on the residuals a cancelled attempt would have shown."""
        residuals = []
        for sample in self._censored:
            prediction, _offset = self._center(
                self._weights(sample.row, self._completed), self._completed, sample
            )
            if prediction is not None:
                residuals.append(sample.log_seconds - math.log(prediction))
        return residuals

    def predict(self, run_id, *, coverage: float | None = None) -> StageDuration | None:
        """Estimate one Stage duration, or ``None`` without a completed sample."""
        query = self.indices.get(run_id)
        if query is None or not self._completed:
            return None
        samples = self._completed
        weights = self._weights(query, samples)
        # Anchor the average on the nearest sample. The offsets then stay small,
        # and a Stage measured once reproduces its own duration exactly instead
        # of paying for an ``exp(log(seconds))`` round trip.
        anchor = samples[max(range(len(samples)), key=weights.__getitem__)]
        center, _offset = self._center(weights, samples, anchor)
        if center is None:
            return None
        upper_residuals = self._residuals()
        lower_residuals = upper_residuals + self._censored_residuals()
        if not upper_residuals or not lower_residuals:
            return StageDuration(center, center, center)
        alpha = 1.0 - (self.coverage if coverage is None else coverage)
        lower = center * math.exp(_order_statistic(lower_residuals, alpha / 2))
        upper = center * math.exp(_order_statistic(upper_residuals, 1 - alpha / 2))
        return StageDuration(
            center=center,
            lower=min(lower, center),
            upper=max(upper, center),
        )
