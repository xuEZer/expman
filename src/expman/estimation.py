"""Experiment-level remaining-time intervals for sequential batches."""

import math
from dataclasses import dataclass
from numbers import Real

from ._time_model import DurationModel, features
from .events import Status


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
        return f"{_clock(self.lower_seconds, upper=False)}～{_clock(self.upper_seconds, upper=True)}"


def _quantile(values, probability):
    position = (len(values) - 1) * probability
    index = int(position)
    a, b = values[index], values[min(index + 1, len(values) - 1)]
    if a == b or position == index:
        return a
    if math.isinf(b):
        return b
    return a + (b - a) * (position - index)


class TimeEstimator:
    def __init__(self, experiments):
        self.experiments = experiments
        self.vectors = features([item.cfg for item in experiments])
        self.indices = {item.run_id: index for index, item in enumerate(experiments)}

    def _observations(self, pending_ids):
        completed, censored, pending, visited = [], [], [], []
        for index, experiment in enumerate(self.experiments):
            attempts, active_elapsed = experiment._timing_snapshot()
            elapsed = sum(item.duration_seconds for item in attempts) + active_elapsed
            if attempts or active_elapsed:
                visited.append(index)
            succeeded = attempts and attempts[-1].status is Status.SUCCEEDED
            if succeeded:
                # A hard exit has unknown elapsed time. Its resumed completion must
                # not enter the training set as an artificially short full experiment.
                if not any(item.error_type == "InterruptedRun" for item in attempts):
                    completed.append((index, max(elapsed, 1e-9)))
            elif elapsed > 0 and experiment.run_id in pending_ids:
                # Terminal failures are not observations of successful completion.
                censored.append((index, elapsed))
            if experiment.run_id in pending_ids and not succeeded:
                pending.append((index, elapsed))
        return completed, censored, pending, visited

    def estimate(self, pending_ids, coverage):
        completed, censored, pending, _ = self._observations(pending_ids)
        if not pending:
            return TimeEstimate(0.0, 0.0, coverage, len(completed), 0)
        if not completed:
            return TimeEstimate(None, None, coverage, 0, len(pending))
        model = DurationModel(self.vectors, completed, censored)
        draws = sorted(model.draws(pending))
        tail = (1 - coverage) / 2
        return TimeEstimate(
            _quantile(draws, tail),
            _quantile(draws, 1 - tail),
            coverage,
            len(completed),
            len(pending),
        )

    def choose(self, candidates, *, remaining_ids):
        """Choose information about the remaining total, with diversity tie breaks."""
        if len(candidates) < 2:
            return candidates[0]
        completed, _, _, visited = self._observations(set(remaining_ids))
        indices = [self.indices[run_id] for run_id in candidates]
        scores = (
            DurationModel(self.vectors, completed, []).priorities(
                [self.indices[run_id] for run_id in remaining_ids]
            )
            if completed
            else {}
        )

        score_scale = max(scores.values(), default=1.0) or 1.0

        def priority(index):
            distance = min(
                (
                    sum(
                        (a - b) ** 2
                        for a, b in zip(
                            self.vectors[index], self.vectors[other], strict=True
                        )
                    )
                    for other in visited
                ),
                default=0.0,
            )
            # Numerical noise must not reorder equivalent categorical candidates.
            return round(scores.get(index, 0.0) / score_scale, 12), distance, -index

        chosen = max(indices, key=priority)
        return self.experiments[chosen].run_id
