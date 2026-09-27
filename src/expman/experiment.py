"""One concrete configuration and the history of its isolated attempts."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol
from uuid import uuid4

from .context import RunContext, _error_message, _name
from .events import Status
from .frozen import freeze
from .metrics import MetricStore
from .pipeline import Pipeline
from .prefix_cache import PrefixCache
from .randomness import RandomStateManager, validate_seed
from .recorders import InMemoryRecorder, Recorder
from .storage import RunStore, Serializer, StorageError, read_record, write_record


class OutputSource(Protocol):
    """Reads an attempt output that the parent process does not keep in memory."""


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

    A batch keeps one of these per attempt for its whole lifetime, so the output
    object is not retained: output_source rereads the authoritative record, and
    every access returns a fresh object.
    """

    run_id: str
    attempt: int
    status: Status
    duration_seconds: float
    _output: Any = None
    error_type: str | None = None
    error_message: str | None = None
    output_source: OutputSource | None = None

    @property
    def output(self) -> Any:
        if self._output is not None or self.output_source is None:
            return self._output
        return self.output_source.load()


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

    Experiment is an internal component: the scheduler executes exactly one
    top-level Stage per worker attempt through run_stage(), which restores the
    completed prefix and the latest checkpoint. Exceptions become summaries;
    BaseException interruptions are recorded and propagated to the caller.
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

    def _output_source(self) -> OutputSource | None:
        """Snapshot source for this run's output when it is re-readable."""
        if not self.pipeline.stages:
            return None
        position = (len(self.pipeline.stages) - 1,)
        if not (self._store.stage_dir(position) / "completed.pkl").exists():
            return None
        return SnapshotOutput(self._store, position)

    def run_stage(
        self, stage_index: int, *, attempt: int, recorder: Recorder | None = None
    ) -> StageResult:
        """Run exactly one top-level stage, restoring the completed prefix."""
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
        except BaseException as error:
            status = Status.FAILED if isinstance(error, Exception) else Status.CANCELLED
            error_type = type(error).__qualname__
            error_message = _error_message(error)
            if not isinstance(error, Exception):
                raise
        return StageResult(
            self.run_id,
            stage_index,
            attempt,
            status,
            perf_counter() - started,
            error_type,
            error_message,
        )
