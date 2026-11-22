"""Reusable Stage groups, ready candidates and their resource estimates."""

from itertools import zip_longest

from expman.core.dependencies import observation, stage_dependencies
from expman.core.events import Status
from expman.estimation.duration_model import DurationModel
from expman.estimation.estimate import valid_seconds
from expman.estimation.memory_model import LocalQuantileEstimator
from expman.estimation.time_model import encode
from expman.resources import limits
from expman.scheduling._planning import Candidate
from expman.settings import settings


class StageWork:
    """Group experiments by declared Stage inputs and price their remaining work."""

    def __init__(self, batch):
        self.batch = batch
        self._groups = {}
        self._encodings = {}
        self._indices = {run_id: index for index, run_id in enumerate(batch.by_id)}
        self._peak_models = {}
        self._duration_models = {}

    def invalidate(self, stage_index):
        """A new observation can change both resource and duration estimates."""
        self._peak_models.pop((stage_index, "peak_kb"), None)
        self._peak_models.pop((stage_index, "gpu_process_peak_kb"), None)
        self._duration_models.pop(stage_index, None)

    def prefix_paths(self, run_id, stage_index):
        """Unique input paths used as features for this Stage's estimates."""
        _seed, prefix = self.group(run_id, stage_index)
        return frozenset(
            path for declaration in prefix for path, _digest in declaration
        )

    def group(self, run_id, stage_index):
        """Match prefix reuse by seed and the declared inputs of each Stage in order."""
        with self.batch._state_lock:
            groups = self._groups.setdefault(run_id, [])
            experiment = self.batch.by_id[run_id]
            config = experiment._frozen
            for index in range(len(groups), stage_index + 1):
                seed, prefix = (
                    groups[-1] if groups else (observation(config, ("seed",)), ())
                )
                stage = experiment.pipeline.stages[index]
                dependencies = tuple(stage_dependencies(stage, config))
                groups.append((seed, (*prefix, dependencies)))
            return groups[stage_index]

    def _completed_groups(self):
        return {
            self.group(row["run_id"], row["stage"])
            for row in self.batch.state.stage_history
            if row["status"] == Status.SUCCEEDED.value
        }

    def remaining_groups(self, queue, running):
        """Return one representative for every not-yet-materialized Stage group."""
        completed = self._completed_groups()
        active = {}
        for run_id, details in running.items():
            stage_index = details.get("stage_index")
            if not isinstance(stage_index, int):
                return None
            active[self.group(run_id, stage_index)] = (run_id, stage_index)
        groups = dict(active)
        # A running attempt still has downstream work. Keep its current group
        # active, then include the stages that will become ready after it ends.
        starts = {run_id: self.batch.state.stage_progress[run_id] for run_id in queue}
        starts.update(
            (run_id, details["stage_index"] + 1) for run_id, details in running.items()
        )
        for run_id, start in starts.items():
            for stage_index in range(start, len(self.batch.pipeline.stages)):
                group = self.group(run_id, stage_index)
                if group not in completed and group not in groups:
                    groups[group] = (run_id, stage_index)
        return groups, set(active)

    def completion(self):
        """Return (completed, total, running, queued) counts per Stage.

        Runs sharing a group contribute once. Completed groups take precedence
        over running groups, and running groups over queued groups.
        """
        state = self.batch.state
        queued = set(state.queue)
        completed = self._completed_groups()
        result = {}
        for index in range(len(self.batch.pipeline.stages)):
            groups, running, pending = set(), set(), set()
            for run_id in self.batch.by_id:
                group = self.group(run_id, index)
                groups.add(group)
                if state.stage_progress.get(run_id) == index:
                    if run_id in state.active_gpu:
                        running.add(group)
                    elif run_id in queued:
                        pending.add(group)
            result[index] = (
                len(groups & completed),
                len(groups),
                len(running - completed),
                len(pending - completed - running),
            )
        return result

    def _encoding(self, stage_index):
        """Encode each run's declared dependency prefix for a Stage input."""
        if stage_index in self._encodings:
            return self._encodings[stage_index]
        configs = []
        for experiment in self.batch.by_id.values():
            cfg = experiment._frozen
            projected = {}
            for path in self.prefix_paths(experiment.run_id, stage_index):
                value = cfg
                try:
                    for key in path:
                        value = value[key]
                except (KeyError, IndexError, TypeError):
                    value = None
                projected[repr(path)] = value
            configs.append(projected)
        result = encode(configs)
        self._encodings[stage_index] = result
        return result

    def peak(self, run_id, stage_index, field, default):
        """Lazily fit one configuration-based peak model per top-level Stage."""
        key = (stage_index, field)
        model = self._peak_models.get(key)
        if model is None:
            model = LocalQuantileEstimator(
                self._encoding(stage_index),
                self._indices,
                default,
                include_zero=False,
            )
            for entry in self.batch.state.stage_history:
                if entry.get("stage") != stage_index or entry.get("reused"):
                    continue
                value = entry.get(field)
                if field == "peak_kb" and entry.get("capped"):
                    value = max(value or 0.0, entry.get("host_retry_floor_kb") or 0.0)
                elif field == "gpu_process_peak_kb":
                    if entry.get("gpu_capped"):
                        value = max(
                            value or 0.0, entry.get("gpu_retry_floor_kb") or 0.0
                        )
                        if not value:
                            continue
                    elif entry.get("status") != Status.SUCCEEDED.value:
                        # Shedding can interrupt a process before it initializes CUDA.
                        continue
                model.record(
                    entry.get("run_id"),
                    value,
                    capped=entry.get(
                        "capped" if field == "peak_kb" else "gpu_capped", False
                    ),
                )
            self._peak_models[key] = model
        return model.estimate(run_id)

    def duration(self, run_id, stage_index, *, coverage=None):
        """Estimate one Stage duration interval from its dependency samples.

        Fit on first use and refit on demand after a Stage result invalidates the
        cache. Training uses recorded attempts. A cancelled attempt contributes
        a lower bound to the lower residual quantile.
        """
        # The timing thread and the dispatch loop both request duration models.
        # Publish a fit under the lock used to invalidate it on Stage completion.
        with self.batch._state_lock:
            if stage_index not in self._duration_models:
                model = DurationModel(
                    self._encoding(stage_index),
                    self._indices,
                    coverage=self.batch.estimate_coverage,
                )
                measured = False
                for entry in self.batch.state.stage_history:
                    if entry.get("stage") != stage_index or entry.get("reused"):
                        continue
                    # Use the same worker-start clock as the running attempt.
                    duration = entry.get("duration")
                    if not valid_seconds(duration) or duration == 0:
                        continue
                    status = entry.get("status")
                    if status == Status.SUCCEEDED.value:
                        measured = True
                        model.record(entry.get("run_id"), duration)
                    elif status == Status.CANCELLED.value:
                        model.record(entry.get("run_id"), duration, censored=True)
                self._duration_models[stage_index] = model if measured else None
            model = self._duration_models[stage_index]
        return None if model is None else model.predict(run_id, coverage=coverage)

    def ready_candidates(self, gpu_capacity_kb, workers):
        """Interleave ready Stages so they share the bounded packing search."""
        by_stage = {}
        for run_id in self.batch.state.queue:
            by_stage.setdefault(self.batch.state.stage_progress[run_id], []).append(
                run_id
            )
        groups = [
            self.candidates(stage, gpu_capacity_kb, by_stage[stage], workers)
            for stage in sorted(by_stage)
        ]
        return [
            candidate
            for row in zip_longest(*groups)
            for candidate in row
            if candidate is not None
        ]

    def candidates(self, stage_index, gpu_capacity_kb, ready, workers):
        """Shortlist idle groups in queue order within retries and first attempts.

        Consider retries first and take at most one representative per group.
        """
        represented = {
            self.group(run_id, stage_index)
            for run_id, worker in workers.items()
            if worker.stage_index == stage_index
        }
        attempts = self.batch.state.stage_attempts
        ordered = sorted(
            ready, key=lambda run_id: f"{run_id}:{stage_index}" not in attempts
        )
        candidates = []
        for run_id in ordered:
            group = self.group(run_id, stage_index)
            if group in represented:
                continue
            represented.add(group)
            host_estimate = self.peak(
                run_id, stage_index, "peak_kb", settings.host_peak_kb_default
            )
            gpu_estimate = self.peak(
                run_id, stage_index, "gpu_process_peak_kb", settings.gpu_peak_kb_default
            )
            candidates.append(
                Candidate(
                    run_id,
                    stage_index,
                    host_estimate,
                    min(gpu_estimate, gpu_capacity_kb),
                    min(limits.cap_kb(gpu_estimate), gpu_capacity_kb),
                    self.duration(run_id, stage_index),
                )
            )
            if len(candidates) == settings.candidates_per_stage:
                break
        return candidates
