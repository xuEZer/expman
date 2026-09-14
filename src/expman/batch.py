"""Durable experiment queues with bounded retries and recovery."""

import math
import warnings
from collections import deque
from dataclasses import asdict
from pathlib import Path
from threading import Event, RLock, Thread
from time import monotonic
from typing import Any
from uuid import uuid4

from .config import load_configs
from .context import _nonnegative_integer
from .devices import configured_devices
from .estimation import TimeEstimate, TimeEstimator, validate_coverage
from .events import Status
from .experiment import AttemptResult, Experiment, ExperimentResult
from .pipeline import Pipeline
from .progress import BatchProgress
from .randomness import validate_seed
from .recorders import InMemoryRecorder, Recorder
from .storage import (
    PickleSerializer,
    RecoveryWarning,
    RunLock,
    Serializer,
    StorageError,
    read_record,
    write_record,
)

TIMING_SAVE_INTERVAL = 5.0


class Batch:
    """A durable experiment collection. Use resume() after interruption.

    Only one process may execute a given output directory at a time.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        cfg: str | Path | dict[str, Any],
        *,
        max_retries: int = 1,
        estimate_coverage: float = 0.8,
        recorder: Recorder | None = None,
        output_dir: str | Path | None = None,
        serializer: Serializer | None = None,
    ) -> None:
        if not isinstance(pipeline, Pipeline):
            raise TypeError("pipeline must be a Pipeline")
        _nonnegative_integer(max_retries, "max_retries")
        if not isinstance(cfg, (str, Path, dict)):
            raise TypeError(
                "cfg must be a YAML path or a concrete configuration dictionary"
            )
        self.estimate_coverage = validate_coverage(estimate_coverage)
        configs = [cfg] if isinstance(cfg, dict) else load_configs(cfg)
        self.devices = configured_devices(configs)
        for config in configs:
            if "seed" not in config:
                raise ValueError("seed is required")
            validate_seed(config["seed"])
        self._active_gpu = {}
        self._stage_progress = {}
        self._stage_attempts = {}
        self._stage_elapsed = {}
        self._stage_history = []
        self._gpu_history = []
        self._gpu_memory = {}
        self._host_memory = {}
        self._gpu_running_info = {}
        self.output_dir = (
            Path("runs") / uuid4().hex if output_dir is None else Path(output_dir)
        ).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._serializer = PickleSerializer() if serializer is None else serializer
        self._signature = pipeline.signature()
        self.experiments = tuple(
            Experiment(
                pipeline,
                config,
                run_id=(run_id := uuid4().hex),
                output_dir=self.output_dir / "experiments" / run_id,
                serializer=self._serializer,
                _metrics_path=self.output_dir / "metrics.sqlite3",
                _cache_root=self.output_dir / "cache",
            )
            for config in configs
        )
        self._stage_progress = {experiment.run_id: 0 for experiment in self.experiments}
        self._stage_elapsed = {
            experiment.run_id: 0.0 for experiment in self.experiments
        }
        self._time_estimator = TimeEstimator(self.experiments)
        self._scheduler = None
        self.max_retries = max_retries
        self.recorder = InMemoryRecorder() if recorder is None else recorder
        self._state_lock = RLock()
        self._started = False
        self._elapsed_seconds = 0.0
        self._session_started = None
        self._last_estimate = None
        self._queue = deque(experiment.run_id for experiment in self.experiments)
        self._save()

    @property
    def elapsed_seconds(self) -> float:
        """Accumulated Batch wall time, excluding time between run invocations."""
        with self._state_lock:
            return self._elapsed_seconds + (
                0.0
                if self._session_started is None
                else monotonic() - self._session_started
            )

    def _save_timing(self):
        with self._state_lock:
            write_record(
                self.output_dir / "timing.pkl",
                {
                    "version": 1,
                    "elapsed_seconds": self.elapsed_seconds,
                    "estimate": None
                    if self._last_estimate is None
                    else asdict(self._last_estimate),
                },
                self._serializer,
            )

    def _timing_loop(self, stopped):
        while not stopped.is_set():
            try:
                try:
                    self._refresh_estimate(self.estimate_coverage)
                finally:
                    self._save_timing()
            except Exception as error:
                warnings.warn(
                    f"could not persist Batch timing: {error}",
                    RecoveryWarning,
                    stacklevel=2,
                )
            stopped.wait(TIMING_SAVE_INTERVAL)

    def _save(self) -> None:
        experiments = []
        for experiment in self.experiments:
            attempts = [
                {
                    "attempt": result.attempt,
                    "status": result.status.value,
                    "duration_seconds": result.duration_seconds,
                    "error_type": result.error_type,
                    "error_message": result.error_message,
                }
                for result in experiment.result.attempts
            ]
            experiments.append({"run_id": experiment.run_id, "attempts": attempts})
        manifest = {
            "version": 1,
            "pipeline": self._signature,
            "max_retries": self.max_retries,
            "experiments": experiments,
            "queue": list(self._queue),
            "active": None,
            "active_gpu": dict(self._active_gpu),
            "stage_progress": dict(self._stage_progress),
            "stage_attempts": dict(self._stage_attempts),
            "stage_elapsed": dict(self._stage_elapsed),
            "stage_history": list(self._stage_history),
            "gpu_history": list(self._gpu_history),
            "estimation": {
                "version": 1,
                "execution": "gpu",
                "coverage": self.estimate_coverage,
            },
        }
        write_record(self.output_dir / "batch.pkl", manifest, self._serializer)
        self._expected_manifest = manifest

    @classmethod
    def resume(
        cls,
        pipeline: Pipeline,
        output_dir: str | Path,
        *,
        recorder: Recorder | None = None,
        serializer: Serializer | None = None,
    ) -> "Batch":
        """Resume trusted local records; the original experiment YAML is not read."""
        if not isinstance(pipeline, Pipeline):
            raise TypeError("pipeline must be a Pipeline")
        self = cls.__new__(cls)
        self.output_dir = Path(output_dir).resolve()
        self._serializer = PickleSerializer() if serializer is None else serializer
        manifest = read_record(self.output_dir / "batch.pkl", self._serializer)
        if (
            not isinstance(manifest, dict)
            or manifest.get("version") != 1
            or not {"pipeline", "max_retries", "experiments", "queue", "active"}
            <= manifest.keys()
            or not isinstance(manifest.get("experiments"), list)
            or not isinstance(manifest.get("queue"), list)
        ):
            raise StorageError("unsupported or invalid batch manifest")
        estimation = manifest.get(
            "estimation", {"version": 1, "execution": "gpu", "coverage": 0.8}
        )
        if (
            not isinstance(estimation, dict)
            or estimation.get("version") != 1
            or estimation.get("execution") != "gpu"
        ):
            raise StorageError("unsupported estimation environment or version")
        self.estimate_coverage = validate_coverage(estimation.get("coverage"))
        self._signature = pipeline.signature()
        if manifest.get("pipeline") != self._signature:
            raise StorageError(
                "pipeline classes, order or available source code changed"
            )
        self.max_retries = manifest["max_retries"]
        _nonnegative_integer(self.max_retries, "max_retries")
        self.recorder = InMemoryRecorder() if recorder is None else recorder
        self._state_lock = RLock()
        self._started = False
        self._stage_progress = dict(manifest.get("stage_progress", {}))
        self._stage_attempts = dict(manifest.get("stage_attempts", {}))
        self._stage_elapsed = dict(manifest.get("stage_elapsed", {}))
        self._stage_history = list(manifest.get("stage_history", []))
        experiments = []
        for entry in manifest["experiments"]:
            run_id = entry["run_id"]
            if (
                not isinstance(run_id, str)
                or len(run_id) != 32
                or any(char not in "0123456789abcdef" for char in run_id)
            ):
                raise StorageError("invalid experiment ID in manifest")
            root = self.output_dir / "experiments" / run_id
            cfg = read_record(root / "config.pkl", self._serializer)
            experiment = Experiment(
                pipeline,
                cfg,
                run_id=run_id,
                output_dir=root,
                serializer=self._serializer,
                _metrics_path=self.output_dir / "metrics.sqlite3",
                _cache_root=self.output_dir / "cache",
                _resume=True,
            )
            for attempt in entry["attempts"]:
                output_source = None
                if attempt["status"] == Status.SUCCEEDED.value and pipeline.stages:
                    position = (len(pipeline.stages) - 1,)
                    if experiment._store.completed(position) is None:
                        raise StorageError(
                            f"successful experiment has no final snapshot: {root}"
                        )
                    output_source = experiment._output_source()
                experiment._attempts.append(
                    AttemptResult(
                        run_id=run_id,
                        attempt=attempt["attempt"],
                        status=Status(attempt["status"]),
                        duration_seconds=attempt["duration_seconds"],
                        output_source=output_source,
                        error_type=attempt["error_type"],
                        error_message=attempt["error_message"],
                    )
                )
            experiments.append(experiment)
        self.experiments = tuple(experiments)
        self._stage_progress = {
            experiment.run_id: int(self._stage_progress.get(experiment.run_id, 0))
            for experiment in self.experiments
        }
        self._stage_elapsed = {
            experiment.run_id: float(self._stage_elapsed.get(experiment.run_id, 0.0))
            for experiment in self.experiments
        }
        self.devices = configured_devices([item.cfg for item in experiments])
        self._active_gpu = {}
        self._gpu_memory = {}
        self._host_memory = {}
        self._gpu_running_info = {}
        history = manifest.get("gpu_history", [])
        attempts_by_id = {
            item.run_id: len(item.result.attempts) for item in experiments
        }
        if not isinstance(history, list) or any(
            not isinstance(item, dict)
            or item.get("run_id") not in attempts_by_id
            or type(item.get("attempt")) is not int
            or not 1 <= item["attempt"] <= attempts_by_id[item["run_id"]]
            or (
                item.get("device") is not None
                and item.get("device") not in self.devices
            )
            or (item.get("uuid") is not None and not isinstance(item.get("uuid"), str))
            or any(
                type(item.get(key)) not in (int, float)
                or not math.isfinite(item[key])
                or item[key] < 0
                for key in ("concurrency", "duration")
            )
            for item in history
        ):
            raise StorageError("invalid persisted GPU observations")
        self._gpu_history = list(history)
        self._time_estimator = TimeEstimator(self.experiments)
        self._scheduler = None
        ids = {experiment.run_id for experiment in self.experiments}
        self._queue = deque(manifest["queue"])
        active = manifest["active"]
        if (
            len(ids) != len(experiments)
            or len(set(self._queue)) != len(self._queue)
            or any(run_id not in ids for run_id in self._queue)
            or active is not None
        ):
            raise StorageError("invalid persisted experiment queue")
        active_gpu = manifest.get("active_gpu", {})
        if not isinstance(active_gpu, dict) or any(
            run_id not in ids
            or run_id in self._queue
            or (device is not None and device not in self.devices)
            for run_id, device in active_gpu.items()
        ):
            raise StorageError("invalid persisted GPU workers")
        interrupted = list(active_gpu)
        for active in reversed(interrupted):
            experiment = next(
                item for item in self.experiments if item.run_id == active
            )
            experiment._attempts.append(
                AttemptResult(
                    run_id=active,
                    attempt=len(experiment._attempts) + 1,
                    status=Status.CANCELLED,
                    duration_seconds=0.0,
                    error_type="InterruptedRun",
                    error_message="previous process ended before recording attempt completion",
                )
            )
            self._queue.appendleft(active)
        self._elapsed_seconds = 0.0
        self._session_started = None
        self._last_estimate = None
        timing_path = self.output_dir / "timing.pkl"
        if timing_path.exists():
            timing = read_record(timing_path, self._serializer)
            elapsed = (
                timing.get("elapsed_seconds") if isinstance(timing, dict) else None
            )
            if (
                not isinstance(timing, dict)
                or timing.get("version") != 1
                or isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(elapsed)
                or elapsed < 0
            ):
                raise StorageError("invalid Batch timing record")
            self._elapsed_seconds = elapsed
            saved = timing.get("estimate")
            if saved is not None:
                try:
                    estimate = TimeEstimate(**saved)
                    validate_coverage(estimate.coverage)
                    if any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value < 0
                        for value in (estimate.lower_seconds, estimate.upper_seconds)
                    ):
                        raise ValueError("invalid estimate bounds")
                    if estimate.lower_seconds > estimate.upper_seconds:
                        raise ValueError("reversed estimate bounds")
                    self._last_estimate = estimate
                except (TypeError, ValueError) as error:
                    raise StorageError("invalid saved time estimate") from error
        self._expected_manifest = manifest
        return self

    @property
    def results(self) -> tuple[ExperimentResult, ...]:
        return tuple(experiment.result for experiment in self.experiments)

    def estimate(self, *, coverage: float | None = None) -> TimeEstimate:
        """Read a remaining-time interval; may be polled while run() executes."""
        level = (
            self.estimate_coverage if coverage is None else validate_coverage(coverage)
        )
        return self._refresh_estimate(level)

    def _display_estimate(self):
        """Read the latest background estimate without fitting on the CLI thread."""
        with self._state_lock:
            pending = set(self._queue) | set(self._active_gpu)
            if not pending:
                return TimeEstimate(0.0, 0.0, self.estimate_coverage, 0, 0)
            if (
                self._last_estimate is not None
                and self._last_estimate.coverage == self.estimate_coverage
            ):
                return self._last_estimate
            return TimeEstimate(None, None, self.estimate_coverage, 0, len(pending))

    def _refresh_estimate(self, level):
        with self._state_lock:
            pending = set(self._queue)
            pending.update(self._active_gpu)
        from .parallel_estimation import estimate_parallel

        scheduler = self._scheduler
        estimate = (
            TimeEstimate(None, None, level, 0, len(pending))
            if scheduler is None
            else estimate_parallel(scheduler, level)
        )
        with self._state_lock:
            if (
                estimate.lower_seconds is not None
                and estimate.upper_seconds is not None
                and math.isfinite(estimate.lower_seconds)
                and math.isfinite(estimate.upper_seconds)
            ):
                self._last_estimate = estimate
            elif (
                self._last_estimate is not None
                and self._last_estimate.coverage == level
            ):
                return self._last_estimate
        return estimate

    def run(
        self, *, progress: bool = True, refresh_interval: float = 1.0
    ) -> tuple[ExperimentResult, ...]:
        """Execute the queue, displaying live progress and ETA on stderr."""
        if self._started:
            raise RuntimeError(
                "a Batch can only run once; use Batch.resume to continue"
            )
        display = BatchProgress(self, enabled=progress, interval=refresh_interval)
        with RunLock(self.output_dir):
            if (
                read_record(self.output_dir / "batch.pkl", self._serializer)
                != self._expected_manifest
            ):
                raise StorageError("batch records changed; reload with Batch.resume")
            self._started = True
            error = None
            self._session_started = monotonic()
            timing_stopped = Event()
            timing_thread = Thread(
                target=self._timing_loop,
                args=(timing_stopped,),
                name="expman-timing",
                daemon=True,
            )
            timing_thread.start()
            try:
                display.start()
                from .scheduling import GpuScheduler

                return GpuScheduler(self).run()
            except BaseException as caught:
                error = caught
                raise
            finally:
                timing_stopped.set()
                timing_thread.join()
                with self._state_lock:
                    self._elapsed_seconds = self.elapsed_seconds
                    self._session_started = None
                display.stop(error)
                try:
                    try:
                        self.estimate()
                    finally:
                        self._save_timing()
                except Exception as timing_error:
                    warnings.warn(
                        f"could not persist Batch timing: {timing_error}",
                        RecoveryWarning,
                        stacklevel=2,
                    )
