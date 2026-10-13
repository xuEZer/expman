"""Models and value objects for remaining-time estimation."""

from expman.estimation.duration_model import DurationModel, StageDuration
from expman.estimation.estimate import (
    UNKNOWN_CLOCK,
    TimeEstimate,
    _clock,
    validate_coverage,
)
from expman.estimation.memory_model import LocalQuantileEstimator, PeakCeiling
from expman.estimation.time_model import Encoding, encode, variable_distance

__all__ = [
    "DurationModel",
    "Encoding",
    "LocalQuantileEstimator",
    "PeakCeiling",
    "StageDuration",
    "TimeEstimate",
    "UNKNOWN_CLOCK",
    "_clock",
    "encode",
    "validate_coverage",
    "variable_distance",
]
