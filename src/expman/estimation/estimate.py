"""Remaining-time interval values used by Batch estimates."""

import math
from dataclasses import dataclass
from numbers import Real
from typing import TypeGuard


def valid_seconds(value: object) -> TypeGuard[float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def validate_coverage(value: object) -> float:
    """A coverage setting is a finite number strictly between 0 and 1."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("estimate coverage must be a finite number between 0 and 1")
    coverage = float(value)
    if not 0 < coverage < 1:
        raise ValueError("estimate coverage must be a finite number between 0 and 1")
    return coverage


def _clock(seconds: object, *, upper: bool) -> str:
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
    ):
        return "??:??:??"
    minutes = math.ceil(seconds / 60) if upper else math.floor(seconds / 60)
    days, remainder = divmod(max(0, minutes), 24 * 60)
    hours, minutes = divmod(remainder, 60)
    return f"{days:02d}:{hours:02d}:{minutes:02d}"


@dataclass(frozen=True)
class TimeEstimate:
    """A remaining-time interval, rendered as ``DD:HH:MM～DD:HH:MM``.

    Each Stage model uses ``coverage`` to select residual quantiles. The
    combined interval reports the completion times of two scheduling scenarios
    at the current concurrency. Its batch-level coverage is uncalibrated.
    """

    lower_seconds: float | None
    upper_seconds: float | None
    coverage: float
    completed_samples: int
    remaining_experiments: int

    @property
    def known(self) -> bool:
        return (
            valid_seconds(self.lower_seconds)
            and valid_seconds(self.upper_seconds)
            and self.lower_seconds <= self.upper_seconds
        )

    def __str__(self) -> str:
        return (
            f"{_clock(self.lower_seconds, upper=False)}"
            f"～{_clock(self.upper_seconds, upper=True)}"
        )
