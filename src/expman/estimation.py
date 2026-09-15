"""Remaining-time interval values used by Batch estimates."""

import math
from dataclasses import dataclass
from numbers import Real


def validate_coverage(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not 0 < value < 1:
        raise ValueError("estimate coverage must be a finite number between 0 and 1")
    return float(value)


def _clock(seconds: float | None, *, upper: bool) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "??:??:??"
    minutes = math.ceil(seconds / 60) if upper else math.floor(seconds / 60)
    days, remainder = divmod(max(0, minutes), 24 * 60)
    hours, minutes = divmod(remainder, 60)
    return f"{days:02d}:{hours:02d}:{minutes:02d}"


@dataclass(frozen=True)
class TimeEstimate:
    """A model-based central prediction interval, not a calibrated guarantee."""

    lower_seconds: float | None
    upper_seconds: float | None
    coverage: float
    completed_samples: int
    remaining_experiments: int
    calibrated: bool = False

    def __str__(self) -> str:
        return _clock(self.upper_seconds, upper=True)
