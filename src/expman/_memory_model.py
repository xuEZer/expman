"""Host memory peak estimates derived from experiment configuration features.

One estimate drives both admission (how much host RAM an attempt is charged) and
the cgroup cap that enforces it. It comes from the same censored log-normal
regression used for durations, read at a high quantile: under-estimating a peak
costs the host, while over-estimating only costs throughput. A run stopped by its
own cap is a right-censored observation, and raises that run's own floor to the
peak it was seen to reach times a margin, so a refused attempt comes back with
room to grow instead of being refused forever.
"""

from dataclasses import dataclass, field

from . import devices
from ._time_model import DurationModel


@dataclass
class PeakEstimator:
    """Per-run host memory peaks for one Batch, in kilobytes."""

    vectors: list
    indices: dict
    default_kb: float = devices.HOST_PEAK_KB_DEFAULT
    quantile: float = devices.PEAK_QUANTILE
    margin: float = devices.PEAK_BUMP_MARGIN
    observed: list = field(default_factory=list)
    censored: list = field(default_factory=list)
    floors: dict = field(default_factory=dict)
    ceiling_kb: float = 0.0
    _model: object = None

    @classmethod
    def from_batch(cls, batch):
        estimator = cls(batch._time_estimator.vectors, batch._time_estimator.indices)
        for entry in batch._gpu_history:
            estimator.record(entry)
        return estimator

    def record(self, entry):
        """Fold one finished attempt's observed peak into the observations."""
        peak = float(entry.get("peak_kb") or 0.0)
        row = self.indices.get(entry.get("run_id"))
        if row is None or peak <= 0:
            return
        self.ceiling_kb = max(self.ceiling_kb, peak)
        if entry.get("capped"):
            # The attempt was stopped at its cap, so the peak it reached is only a
            # lower bound on what it needed.
            self.censored.append((row, peak))
            self.floors[entry["run_id"]] = max(
                self.floors.get(entry["run_id"], 0.0), peak * self.margin
            )
        else:
            self.observed.append((row, peak))
        self._model = None

    def estimate_kb(self, run_id):
        """Peak reserved for one run: regression upper quantile, floored by bumps.

        A configuration with no observation at all falls back to the batch default,
        which is the cold start for a whole Batch and for a run whose row is
        unknown.
        """
        floor = self.floors.get(run_id, 0.0)
        row = self.indices.get(run_id)
        if row is None or not self.observed:
            return max(self.default_kb, floor)
        if self._model is None:
            self._model = DurationModel(self.vectors, self.observed, self.censored)
        return max(self._model.upper_quantile(row, self.quantile), floor)
