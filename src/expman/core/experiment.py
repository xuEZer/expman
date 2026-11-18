"""One concrete configuration and the history of its isolated attempts."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol
from uuid import uuid4

from expman.core.context import RunContext, _error_message, _name
from expman.core.events import Status
from expman.core.frozen import freeze
from expman.core.pipeline import Pipeline
from expman.core.randomness import RandomStateManager, validate_seed
from expman.core.recorders import InMemoryRecorder, Recorder
from expman.storage.base import (
    Serializer,
    StorageError,
    read_record,
    write_record,
)
from expman.storage.metrics import MetricStore
from expman.storage.prefix_cache import PrefixCache
from expman.storage.run import RunStore


class OutputSource(Protocol):
    """Reads an attempt output that the parent process does not keep in memory."""

    def load(self) -> Any: ...


@dataclass(frozen=True, eq=False)
class SnapshotOutput:
    """Attempt output kept in the final stage snapshot of its run."""

    store: RunStore
    position: tuple[int, ...]

    def load(self) -> Any:
        record = self.store.completed(self.position)
        if record is None:
            raise StorageError(f"missing stage snapshot: {self.position}")
        return record["output"]


@dataclass(frozen=True)
class AttemptResult:
    """Attempt summary; the output is read from its record on demand.

    Batch retains the summaries. A successful run with stages reads its final
    snapshot each time the caller requests the output.
    """

    run_id: str
    attempt: int
    status: Status
    duration_seconds: float
    error_type: str | None = None
    error_message: str | None = None
    output_source: OutputSource | None = None

    @property
    def output(self) -> Any:
        return None if self.output_source is None else self.output_source.load()


@dataclass(frozen=True)
class StageResult:
    """One top-level Stage execution, used by the Stage scheduler."""

    run_id: str
    stage_index: int
    attempt: int
    status: Status
    duration_seconds: float
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class ExperimentResult:
    run_id: str
    attempts: tuple[AttemptResult, ...]

    @property
    def status(self) -> Status:
        return self.attempts[-1].status if self.attempts else Status.PENDING

    @property
    def output(self) -> Any:
        return self.attempts[-1].output if self.attempts else None

    @property
    def duration_seconds(self) -> float:
        return sum(attempt.duration_seconds for attempt in self.attempts)


class Experiment:
    """A stable run identity with durable positional stage snapshots.

    A worker requests one top-level Stage through run_stage(). It restores
    the completed prefix and reuses or computes the target. Execution
    exceptions become failed StageResults. Interruptions propagate to the
    worker, which reports their outcome to the scheduler when possible.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        cfg: dict[str, Any],
        *,
        run_id: str | None = None,
        output_dir: str | Path | None = None,
        serializer: Serializer | None = None,
        _resume: bool = False,
        _metrics_path: Path | None = None,
        _cache_root: Path | None = None,
    ) -> None:
        if not isinstance(pipeline, Pipeline):
            raise TypeError("pipeline must be a Pipeline")
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be a dictionary")
        self.pipeline = pipeline
        self._cfg = deepcopy(cfg)
        if "seed" not in self._cfg:
            raise ValueError("seed is required")
        validate_seed(self._cfg["seed"])
        self._frozen = freeze(self._cfg)
        self._run_id = uuid4().hex if run_id is None else run_id
        _name(self._run_id, "run_id")
        self._attempts: list[AttemptResult] = []
        self.output_dir = (
            Path("runs") / self.run_id if output_dir is None else Path(output_dir)
        )
        self._metrics = MetricStore(
            self.output_dir / "metrics.sqlite3"
            if _metrics_path is None
            else _metrics_path
        )
        self._store = RunStore(self.output_dir, serializer)
        self._cache_root = _cache_root
        if _cache_root is not None:
            self._store.shared = PrefixCache(
                _cache_root,
                pipeline.signature(),
                self._cfg["seed"],
                self._store.serializer,
            )
            self._store.metrics = self._metrics
            self._store.run_id = self._run_id
        if not _resume:
            self.output_dir.mkdir(parents=True, exist_ok=False)
            write_record(
                self.output_dir / "config.pkl", self._cfg, self._store.serializer
            )

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def cfg(self) -> dict[str, Any]:
        return deepcopy(self._cfg)

    @property
    def result(self) -> ExperimentResult:
        return ExperimentResult(self.run_id, tuple(self._attempts))

    def record_attempt(
        self,
        status: Status,
        duration_seconds: float,
        *,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Append one finished attempt; successful output stays in its snapshot."""
        source = None
        if status is Status.SUCCEEDED and self.pipeline.stages:
            position = (len(self.pipeline.stages) - 1,)
            if not self._store.has_completed(position):
                raise StorageError(
                    f"successful experiment has no final snapshot: {self.output_dir}"
                )
            source = SnapshotOutput(self._store, position)
        self._attempts.append(
            AttemptResult(
                self.run_id,
                len(self._attempts) + 1,
                status,
                duration_seconds,
                error_type=error_type,
                error_message=error_message,
                output_source=source,
            )
        )

    def run_stage(
        self, stage_index: int, *, attempt: int, recorder: Recorder | None = None
    ) -> StageResult:
        """Restore the required prefix and reuse or execute the requested Stage."""
        if not 0 <= stage_index < len(self.pipeline.stages):
            raise ValueError("stage_index must identify a top-level Pipeline stage")
        for index in range(stage_index):
            if not self._store.has_completed((index,)):
                raise StorageError(
                    "a Stage worker may only execute its scheduled top-level Stage"
                )
        started = perf_counter()
        status = Status.SUCCEEDED
        error_type = error_message = None
        try:
            rng = RandomStateManager(self._cfg["seed"])
            initial = self.output_dir / "rng_initial.pkl"
            if initial.exists():
                rng.restore(read_record(initial, self._store.serializer))
            else:
                rng.seed()
                write_record(initial, rng.capture(), self._store.serializer)
            self._store.rng = rng
            ctx = RunContext(
                run_id=self.run_id,
                recorder=InMemoryRecorder() if recorder is None else recorder,
                cfg=self._frozen,
                attempt=attempt,
                _store=self._store,
                _metrics=self._metrics,
            )
            with ctx.observe(self.pipeline.name, kind="experiment") as context:
                self.pipeline.run(ctx=context, stop_after=stage_index)
        except Exception as error:
            status = Status.FAILED
            error_type = type(error).__qualname__
            error_message = _error_message(error)
        return StageResult(
            self.run_id,
            stage_index,
            attempt,
            status,
            perf_counter() - started,
            error_type,
            error_message,
        )
