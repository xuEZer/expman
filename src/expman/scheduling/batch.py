"""Create, run and reopen a durable batch of experiment configurations."""

import warnings
from pathlib import Path
from threading import Event, RLock, Thread
from time import monotonic
from typing import Any
from uuid import uuid4

from expman.config import configured_devices
from expman.core.context import _nonnegative_integer
from expman.core.events import Status
from expman.core.experiment import AttemptResult, Experiment, ExperimentResult
from expman.core.pipeline import Pipeline
from expman.core.randomness import validate_seed
from expman.core.recorders import InMemoryRecorder
from expman.estimation.estimate import TimeEstimate, valid_seconds, validate_coverage
from expman.observation.cleanup import clean_locked
from expman.process_control import stop_on_sigterm
from expman.scheduling.environment import validate as validate_environment
from expman.scheduling.state import BatchState, LiveObservations
from expman.settings import settings
from expman.storage.base import RecoveryWarning, StorageError, run_lock
from expman.storage.batch import BatchStore


def _attempt_record(result: AttemptResult) -> dict:
    return {
        "attempt": result.attempt,
        "status": result.status.value,
        "duration_seconds": result.duration_seconds,
        "error_type": result.error_type,
        "error_message": result.error_message,
    }


class _Batch:
    """The CLI's execution state for one pipeline and its runs."""

    def __init__(
        self,
        pipeline: Pipeline,
        store: BatchStore,
        retries: object,
        coverage: object,
        pipeline_spec: str,
    ) -> None:
        self.pipeline = pipeline
        self.store = store
        self.output_dir = self.store.directory
        self.max_retries = _nonnegative_integer(retries, "max_retries")
        self.estimate_coverage = validate_coverage(coverage)
        self._pipeline_spec = pipeline_spec
        self._signature = pipeline.signature()
        self.by_id: dict[str, Experiment] = {}
        self.devices: tuple[int, ...]
        self.state: BatchState
        self.recorder = InMemoryRecorder()
        self.live = LiveObservations()
        self._state_lock = RLock()
        self._scheduler = None
        self._elapsed_seconds = 0.0
        self._session_started = None
        self._last_estimate = None

    @classmethod
    def create(
        cls,
        pipeline: Pipeline,
        configs: list[dict[str, Any]],
        *,
        output_dir: Path,
        pipeline_spec: str,
    ) -> "_Batch":
        validate_environment()
        batch = cls(
            pipeline,
            BatchStore(output_dir.parent.resolve() / output_dir.name),
            settings.retries,
            settings.coverage,
            pipeline_spec,
        )
        batch.devices = configured_devices(configs)
        for config in configs:
            if "seed" not in config:
                raise ValueError("seed is required")
            validate_seed(config["seed"])
        batch.output_dir.mkdir(parents=True, exist_ok=False)
        for config in configs:
            run_id = uuid4().hex
            batch.by_id[run_id] = batch._experiment(run_id, config, resume=False)
        batch.state = BatchState.new(batch.by_id)
        batch._save()
        return batch

    def _experiment(self, run_id, cfg, *, resume):
        return Experiment(
            self.pipeline,
            cfg,
            run_id=run_id,
            output_dir=self.output_dir / "experiments" / run_id,
            serializer=self.store.serializer,
            _metrics_path=self.output_dir / "metrics.sqlite3",
            _cache_root=self.output_dir / "cache",
            _resume=resume,
        )

    @property
    def experiments(self) -> tuple[Experiment, ...]:
        return tuple(self.by_id.values())

    @property
    def results(self) -> tuple[ExperimentResult, ...]:
        return tuple(run.result for run in self.by_id.values())

    def record_attempt(
        self, run_id: str, status: Status, *, error_type=None, error_message=None
    ) -> None:
        """Account for each worker duration exactly once, including after retries."""
        self.by_id[run_id].record_attempt(
            status,
            self.state.stage_elapsed[run_id],
            error_type=error_type,
            error_message=error_message,
        )
        self.state.stage_elapsed[run_id] = 0.0

    def _save(self) -> None:
        with self._state_lock:
            self._revision = self.store.save_manifest(
                {
                    "pipeline": self._signature,
                    "pipeline_spec": self._pipeline_spec,
                    "max_retries": self.max_retries,
                    "estimate_coverage": self.estimate_coverage,
                    "experiments": [
                        {
                            "run_id": run.run_id,
                            "attempts": [
                                _attempt_record(a) for a in run.result.attempts
                            ],
                        }
                        for run in self.by_id.values()
                    ],
                    **self.state.to_record(),
                }
            )

    @classmethod
    def resume(
        cls,
        pipeline: Pipeline,
        store: BatchStore,
        manifest: dict,
    ) -> "_Batch":
        try:
            batch = cls(
                pipeline,
                store,
                manifest["max_retries"],
                manifest["estimate_coverage"],
                manifest["pipeline_spec"],
            )
            if manifest["pipeline"] != batch._signature:
                raise StorageError(
                    "pipeline classes, order or available source code changed"
                )
            batch._revision = manifest["revision"]
            for entry in manifest["experiments"]:
                run_id = entry["run_id"]
                if (
                    not isinstance(run_id, str)
                    or len(run_id) != 32
                    or any(c not in "0123456789abcdef" for c in run_id)
                    or run_id in batch.by_id
                ):
                    raise StorageError("invalid or duplicate experiment ID in manifest")
                run = batch._experiment(run_id, store.load_config(run_id), resume=True)
                batch.by_id[run_id] = run
                for index, saved in enumerate(entry["attempts"], start=1):
                    if saved["attempt"] != index or not valid_seconds(
                        saved["duration_seconds"]
                    ):
                        raise StorageError("invalid attempt in batch manifest")
                    run.record_attempt(
                        Status(saved["status"]),
                        saved["duration_seconds"],
                        error_type=saved["error_type"],
                        error_message=saved["error_message"],
                    )
            batch.devices = configured_devices(
                [run.cfg for run in batch.by_id.values()]
            )
            batch.state = BatchState.from_record(
                manifest, set(batch.by_id), batch.devices, len(pipeline.stages)
            )
        except (KeyError, TypeError, ValueError) as error:
            raise StorageError("invalid batch manifest") from error
        for run_id in batch.state.recover_active():
            batch.record_attempt(
                run_id,
                Status.CANCELLED,
                error_type="InterruptedRun",
                error_message="previous process ended before recording attempt completion",
            )
        batch._elapsed_seconds, batch._last_estimate = store.read_timing() or (
            0.0,
            None,
        )
        return batch

    @property
    def elapsed_seconds(self) -> float:
        with self._state_lock:
            if self._session_started is None:
                return self._elapsed_seconds
            return self._elapsed_seconds + monotonic() - self._session_started

    def estimate(self, *, coverage: float | None = None) -> TimeEstimate:
        from expman.scheduling.parallel_estimation import estimate_parallel

        level = (
            self.estimate_coverage if coverage is None else validate_coverage(coverage)
        )
        with self._state_lock:
            scheduler = self._scheduler
            pending = len(self.state.queue) + len(self.state.active_gpu)
        if scheduler is None:
            bound = None if pending else 0.0
            estimate = TimeEstimate(bound, bound, level, 0, pending)
        else:
            estimate = estimate_parallel(scheduler, level)
        if level == self.estimate_coverage:
            with self._state_lock:
                self._last_estimate = estimate
        return estimate

    def _snapshot_estimate(self):
        """Get the latest default-coverage estimate for a live snapshot."""
        with self._state_lock:
            pending = len(self.state.queue) + len(self.state.active_gpu)
            if not pending:
                return TimeEstimate(0.0, 0.0, self.estimate_coverage, 0, 0)
            if (
                self._last_estimate is not None
                and self._last_estimate.coverage == self.estimate_coverage
            ):
                return self._last_estimate
            return TimeEstimate(None, None, self.estimate_coverage, 0, pending)

    def _save_timing(self):
        with self._state_lock:
            self.store.save_timing(self.elapsed_seconds, self._last_estimate)

    def _update_timing(self):
        try:
            try:
                self.estimate()
            finally:
                self._save_timing()
        except Exception as error:
            warnings.warn(
                f"could not persist Batch timing: {error}",
                RecoveryWarning,
                stacklevel=2,
            )

    def _timing_loop(self, stopped):
        while not stopped.wait(settings.timing_save_interval):
            self._update_timing()

    def run(self) -> tuple[ExperimentResult, ...]:
        """Execute the batch while accepting CLI and dashboard stop requests."""
        from expman.scheduling.scheduler import GpuScheduler

        with run_lock(self.output_dir), stop_on_sigterm(self.output_dir):
            if self.store.read_manifest()["revision"] != self._revision:
                raise StorageError("batch records changed; retry expman resume")
            if self._elapsed_seconds > 0:
                self.state.session_marks.append(round(self._elapsed_seconds, 2))
                self._save()
            self._session_started = monotonic()
            stopped = Event()
            timer = Thread(
                target=self._timing_loop,
                args=(stopped,),
                name="expman-timing",
                daemon=True,
            )
            timer.start()
            live = self._start_live()
            try:
                if not self.pipeline.stages:
                    with self._state_lock:
                        while self.state.queue:
                            self.record_attempt(
                                self.state.queue.popleft(), Status.SUCCEEDED
                            )
                        self._save()
                    return self.results
                results = GpuScheduler(self).run()
                if all(run.status is Status.SUCCEEDED for run in results):
                    clean_locked(self.output_dir, self.store.serializer)
                return results
            finally:
                stopped.set()
                timer.join()
                with self._state_lock:
                    self._elapsed_seconds = self.elapsed_seconds
                    self._session_started = None
                self._update_timing()
                if live is not None:
                    live.close()

    def _start_live(self):
        from expman.observation.live import LivePublisher, live_enabled

        if not live_enabled():
            return None
        try:
            return LivePublisher(self).start()
        except Exception as error:
            warnings.warn(
                f"could not publish live dashboard state: {error}",
                RecoveryWarning,
                stacklevel=2,
            )
            return None
