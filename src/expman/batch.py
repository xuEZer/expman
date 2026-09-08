"""Durable sequential experiment queues with bounded retries and recovery."""

import gc
import warnings
from collections import deque
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import load_configs
from .context import _nonnegative_integer
from .events import Status
from .experiment import AttemptResult, Experiment, ExperimentResult
from .pipeline import Pipeline
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
        configs = [cfg] if isinstance(cfg, dict) else load_configs(cfg)
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
            )
            for config in configs
        )
        self.max_retries = max_retries
        self.recorder = InMemoryRecorder() if recorder is None else recorder
        self._started = False
        self._queue = deque(experiment.run_id for experiment in self.experiments)
        self._active = None
        self._save()

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
            "active": self._active,
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
        self._signature = pipeline.signature()
        if manifest.get("pipeline") != self._signature:
            raise StorageError(
                "pipeline classes, order or available source code changed"
            )
        self.max_retries = manifest["max_retries"]
        _nonnegative_integer(self.max_retries, "max_retries")
        self.recorder = InMemoryRecorder() if recorder is None else recorder
        self._started = False
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
                _resume=True,
            )
            for attempt in entry["attempts"]:
                output = None
                if attempt["status"] == Status.SUCCEEDED.value and pipeline.stages:
                    completed = experiment._store.completed((len(pipeline.stages) - 1,))
                    if completed is None:
                        raise StorageError(
                            f"successful experiment has no final snapshot: {root}"
                        )
                    output = completed["output"]
                experiment._attempts.append(
                    AttemptResult(
                        run_id=run_id,
                        attempt=attempt["attempt"],
                        status=Status(attempt["status"]),
                        duration_seconds=attempt["duration_seconds"],
                        output=output,
                        error_type=attempt["error_type"],
                        error_message=attempt["error_message"],
                    )
                )
            experiments.append(experiment)
        self.experiments = tuple(experiments)
        ids = {experiment.run_id for experiment in self.experiments}
        self._queue = deque(manifest["queue"])
        active = manifest["active"]
        if (
            len(ids) != len(experiments)
            or len(set(self._queue)) != len(self._queue)
            or any(run_id not in ids for run_id in self._queue)
            or (active is not None and (active not in ids or active in self._queue))
        ):
            raise StorageError("invalid persisted experiment queue")
        if active is not None:
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
        self._active = None
        self._expected_manifest = manifest
        return self

    @property
    def results(self) -> tuple[ExperimentResult, ...]:
        return tuple(experiment.result for experiment in self.experiments)

    def run(self) -> tuple[ExperimentResult, ...]:
        if self._started:
            raise RuntimeError(
                "a Batch can only run once; use Batch.resume to continue"
            )
        with RunLock(self.output_dir):
            if (
                read_record(self.output_dir / "batch.pkl", self._serializer)
                != self._expected_manifest
            ):
                raise StorageError("batch records changed; reload with Batch.resume")
            self._started = True
            return self._run_queue()

    def _run_queue(self) -> tuple[ExperimentResult, ...]:
        experiments = {experiment.run_id: experiment for experiment in self.experiments}
        while self._queue:
            self._active = self._queue.popleft()
            experiment = experiments[self._active]
            self._save()
            try:
                result = experiment.run(recorder=self.recorder)
            except BaseException:
                self._queue.appendleft(self._active)
                self._active = None
                try:
                    self._save()
                except Exception as error:
                    warnings.warn(
                        f"could not record interruption: {error}",
                        RecoveryWarning,
                        stacklevel=2,
                    )
                raise
            if result.status is Status.FAILED:
                gc.collect()
                failures = sum(
                    item.status is Status.FAILED for item in experiment.result.attempts
                )
                if failures <= self.max_retries:
                    self._queue.append(experiment.run_id)
            self._active = None
            self._save()
        return self.results
