"""Bounded local estimates for per-Stage resource peaks.

Resource admission needs a conservative estimate, but a sparse high-dimensional
regression can turn modest uncertainty in log space into an arbitrarily large
number after exponentiation. The scheduler therefore uses nearby, finite
observations from the same Stage. A capped cgroup sample is a lower bound, so it
receives one fixed bump; it is never extrapolated into an unbounded tail.
"""

from dataclasses import dataclass, field
from math import ceil, isfinite

from . import devices

LOCAL_MIN_SAMPLES = 4
LOCAL_NEIGHBORS = 8


@dataclass
class LocalQuantileEstimator:
    """A bounded high-quantile estimate from nearby configuration samples.

    ``vectors`` are feature rows for all candidate runs and ``indices`` maps a
    run id to its row. With only a few samples the Stage-wide observations are
    safer than a spurious feature split. Once enough samples exist, only the
    closest configurations participate. In either case the returned value is an
    empirical order statistic, bounded by the largest finite adjusted sample.
    """

    vectors: list
    indices: dict
    default: float
    quantile: float = devices.PEAK_QUANTILE
    margin: float = devices.PEAK_BUMP_MARGIN
    include_zero: bool = False
    observations: list[tuple[int, float]] = field(default_factory=list)
    floors: dict = field(default_factory=dict)

    def record(self, run_id, value, *, capped=False):
        """Add one finite observation, applying one bounded cap retry bump."""
        row = self.indices.get(run_id)
        if row is None or not isinstance(value, (int, float)):
            return
        value = float(value)
        if not isfinite(value) or value < 0 or (value == 0 and not self.include_zero):
            return
        adjusted = value * self.margin if capped else value
        self.observations.append((row, adjusted))
        if capped:
            self.floors[run_id] = max(self.floors.get(run_id, 0.0), adjusted)

    @staticmethod
    def _distance(left, right):
        # Feature rows start with an intercept, which cannot distinguish runs.
        return sum((a - b) ** 2 for a, b in zip(left[1:], right[1:], strict=True))

    def _nearby(self, row):
        if len(self.observations) < LOCAL_MIN_SAMPLES:
            return [value for _, value in self.observations]
        closest = sorted(
            self.observations,
            key=lambda item: self._distance(row, self.vectors[item[0]]),
        )[:LOCAL_NEIGHBORS]
        return [value for _, value in closest]

    def estimate(self, run_id):
        """Return a finite conservative local estimate for ``run_id``."""
        floor = self.floors.get(run_id, 0.0)
        index = self.indices.get(run_id)
        if index is None or not self.observations:
            return max(self.default, floor)
        values = sorted(self._nearby(self.vectors[index]))
        # Higher order statistic: never interpolate below a sample when the
        # requested percentile falls between two observations.
        position = max(0, min(len(values) - 1, ceil(self.quantile * len(values)) - 1))
        return max(values[position], floor)


@dataclass
class PeakCeiling:
    """Largest observed whole-attempt host peak for cold-start admission."""

    ceiling_kb: float = 0.0

    @classmethod
    def from_history(cls, history):
        ceiling = cls()
        for entry in history:
            ceiling.record(entry)
        return ceiling

    def record(self, entry):
        peak = entry.get("peak_kb")
        if not isinstance(peak, (int, float)) or not isfinite(peak) or peak <= 0:
            return
        adjusted = (
            float(peak) * devices.PEAK_BUMP_MARGIN
            if entry.get("capped")
            else float(peak)
        )
        self.ceiling_kb = max(self.ceiling_kb, adjusted)
