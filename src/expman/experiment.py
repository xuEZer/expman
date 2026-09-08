"""One concrete configuration and the history of its isolated attempts."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any
from uuid import uuid4

from .context import RunContext, _error_message, _name
from .events import Status
from .frozen import freeze
from .metrics import MetricStore
from .pipeline import Pipeline
from .recorders import InMemoryRecorder, Recorder
from .storage import RunStore, Serializer, write_record


@dataclass(frozen=True)
class AttemptResult:
    run_id: str
    attempt: int
    status: Status
    duration_seconds: float
    output: Any = None
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

    run() restores prior progress and executes one attempt. Exceptions become summaries;
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
    ) -> None:
        if not isinstance(pipeline, Pipeline):
            raise TypeError("pipeline must be a Pipeline")
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be a dictionary")
        self.pipeline = pipeline
        self._cfg = deepcopy(cfg)
        freeze(self._cfg)
        self._run_id = uuid4().hex if run_id is None else run_id
        _name(self._run_id, "run_id")
        self._attempts: list[AttemptResult] = []
        self._timing_lock = RLock()
        self._active_started: float | None = None
        self.output_dir = (
            Path("runs") / self.run_id if output_dir is None else Path(output_dir)
        )
        self._metrics = MetricStore(
            self.output_dir / "metrics.sqlite3"
            if _metrics_path is None
            else _metrics_path
        )
        self._store = RunStore(self.output_dir, serializer)
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

    def _timing_snapshot(self) -> tuple[tuple[AttemptResult, ...], float]:
        with self._timing_lock:
            elapsed = (
                0.0
                if self._active_started is None
                else max(0.0, perf_counter() - self._active_started)
            )
            return tuple(self._attempts), elapsed

    def run(self, *, recorder: Recorder | None = None) -> AttemptResult:
        """Execute once, restoring completed stages and the latest checkpoint."""
        attempt = len(self._attempts) + 1
        started = perf_counter()
        with self._timing_lock:
            self._active_started = started
        status = Status.SUCCEEDED
        output = None
        error_type = None
        error_message = None
        try:
            ctx = RunContext(
                run_id=self.run_id,
                recorder=InMemoryRecorder() if recorder is None else recorder,
                cfg=deepcopy(self._cfg),
                attempt=attempt,
                _store=self._store,
                _metrics=self._metrics,
            )
            with ctx.observe(self.pipeline.name, kind="experiment") as context:
                output = self.pipeline.run(ctx=context)
        except BaseException as error:
            status = Status.FAILED if isinstance(error, Exception) else Status.CANCELLED
            error_type = type(error).__qualname__
            error_message = _error_message(error)
            if not isinstance(error, Exception):
                raise
        finally:
            result = AttemptResult(
                run_id=self.run_id,
                attempt=attempt,
                status=status,
                duration_seconds=perf_counter() - started,
                output=output,
                error_type=error_type,
                error_message=error_message,
            )
            with self._timing_lock:
                self._attempts.append(result)
                self._active_started = None
        return result
