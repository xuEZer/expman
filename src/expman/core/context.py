"""Run-scoped services and nested execution observation."""

import logging
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from numbers import Real
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from expman.core.events import Event, ExecutionEvent, MetricEvent, ProgressEvent, Status
from expman.core.frozen import freeze
from expman.core.recorders import InMemoryRecorder, Recorder
from expman.storage.base import Checkpoint, RunStore
from expman.storage.metrics import MetricStore, flatten_metrics

_logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(UTC)


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _name(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _error_message(error: BaseException) -> str:
    try:
        return str(error)
    except Exception:
        return "<exception message unavailable>"


@dataclass(frozen=True)
class RunContext:
    """Shared recorder and identity, with an immutable execution scope.

    Each call to observe creates a child scope while retaining the run ID and
    recorder. Ordinary recorder exceptions are logged and suppressed so an
    unavailable monitoring sink does not change the business outcome.
    """

    run_id: str = field(default_factory=lambda: uuid4().hex)
    recorder: Recorder = field(default_factory=InMemoryRecorder)
    cfg: Mapping[str, Any] = field(default_factory=dict, kw_only=True)
    stage_id: int | None = field(default=None, kw_only=True)
    _metrics: MetricStore | None = field(default=None, kw_only=True, repr=False)
    _store: RunStore | None = field(default=None, kw_only=True, repr=False)
    _stage_path: tuple[int, ...] = field(default=(), kw_only=True, repr=False)
    _checkpoint: Checkpoint | None = field(default=None, kw_only=True, repr=False)
    _checkpoint_record: dict | None = field(default=None, kw_only=True, repr=False)
    _pipeline_calls: dict = field(default_factory=dict, kw_only=True, repr=False)
    _explicit_checkpoint_used: bool = field(default=False, kw_only=True, repr=False)
    attempt: int = field(default=1, kw_only=True)
    _execution_id: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _name(self.run_id, "run_id")
        _nonnegative_integer(self.attempt, "attempt")
        if self.attempt == 0:
            raise ValueError("attempt must be at least 1")
        if not isinstance(self.cfg, Mapping):
            raise TypeError("cfg must be a mapping")
        object.__setattr__(self, "cfg", freeze(self.cfg))

    @property
    def execution_id(self) -> str | None:
        return self._execution_id

    def emit(self, event: Event) -> None:
        try:
            self.recorder.record(event)
        except Exception:
            _logger.warning("Recorder failed for run %s", self.run_id, exc_info=True)

    @contextmanager
    def observe(
        self,
        name: str,
        *,
        kind: Literal["experiment", "pipeline", "stage"],
        reused: bool = False,
    ) -> Iterator["RunContext"]:
        """Record start and finish, preserving the original business exception."""
        _name(name, "execution name")
        if kind not in ("experiment", "pipeline", "stage"):
            raise ValueError("kind must be 'experiment', 'pipeline' or 'stage'")
        child = replace(self, _execution_id=uuid4().hex)
        execution_id = child.execution_id
        assert execution_id is not None
        self.emit(
            ExecutionEvent(
                run_id=self.run_id,
                execution_id=execution_id,
                parent_id=self.execution_id,
                kind=kind,
                name=name,
                status=Status.RUNNING,
                timestamp=_now(),
                attempt=self.attempt,
                stage_id=self.stage_id,
                reused=reused,
            )
        )
        # Exclude this execution's start/end recorder calls from its duration.
        started = perf_counter()
        status = Status.SUCCEEDED
        error_type = None
        error_message = None
        try:
            yield child
        except BaseException as error:
            status = (
                Status.CANCELLED if not isinstance(error, Exception) else Status.FAILED
            )
            error_type = type(error).__qualname__
            error_message = _error_message(error)
            raise
        finally:
            duration = perf_counter() - started
            self.emit(
                ExecutionEvent(
                    run_id=self.run_id,
                    execution_id=execution_id,
                    parent_id=self.execution_id,
                    kind=kind,
                    name=name,
                    status=status,
                    timestamp=_now(),
                    duration_seconds=duration,
                    error_type=error_type,
                    error_message=error_message,
                    attempt=self.attempt,
                    stage_id=self.stage_id,
                    reused=reused,
                )
            )

    def log_metrics(
        self, metrics: Mapping[str, Any], *, step: int | None = None
    ) -> None:
        """Commit nested scalar metrics; storage errors fail the current stage."""
        if self._metrics is None or not self._stage_path:
            raise RuntimeError(
                "log_metrics requires a stage managed by Experiment or Batch"
            )
        if step is not None:
            _nonnegative_integer(step, "step")
            if step > 2**63 - 1:
                raise ValueError("step must fit a SQLite signed 64-bit integer")
        rows = flatten_metrics(metrics)
        self._metrics.log(self.run_id, self._stage_path, step, self.attempt, rows)

    def report_metric(
        self, name: str, value: float, *, step: int | None = None
    ) -> None:
        """Report a finite scalar metric within the current execution."""
        _name(name, "metric name")
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
        ):
            raise ValueError("metric value must be a finite real number")
        if step is not None:
            _nonnegative_integer(step, "step")
        self.emit(
            MetricEvent(
                run_id=self.run_id,
                execution_id=self.execution_id,
                name=name,
                value=float(value),
                step=step,
                timestamp=_now(),
                attempt=self.attempt,
                stage_id=self.stage_id,
            )
        )

    def report_progress(
        self, completed: int, *, total: int | None = None, unit: str = "step"
    ) -> None:
        """Report an absolute work count; total may be unknown."""
        _nonnegative_integer(completed, "completed")
        if total is not None:
            _nonnegative_integer(total, "total")
            if completed > total:
                raise ValueError("completed must not exceed total")
        _name(unit, "unit")
        self.emit(
            ProgressEvent(
                run_id=self.run_id,
                execution_id=self.execution_id,
                completed=completed,
                total=total,
                unit=unit,
                timestamp=_now(),
                attempt=self.attempt,
                stage_id=self.stage_id,
            )
        )

    @property
    def checkpoint_state(self) -> Any:
        """The state carried by the latest checkpoint, or None from scratch.

        A state saved through ``checkpoint()`` with a non-dictionary value is
        unwrapped transparently; dictionary states are returned as-is.
        """
        record = self._checkpoint_record
        if record is None:
            return None
        state = record.get("state")
        if isinstance(state, dict) and set(state) == {"__payload__"}:
            return state["__payload__"]
        return state

    def resume_range(self, start: int, stop: int):
        """Iterate from ``start`` to ``stop``, resuming from the checkpoint step.

        If a checkpoint exists at step ``s``, iteration begins at
        ``max(start, s)``; otherwise it begins at ``start``.  An empty iterator
        is returned when the checkpoint is already at or beyond ``stop``.
        """
        start = _nonnegative_integer(start, "resume_range start")
        stop = _nonnegative_integer(stop, "resume_range stop")
        if start > stop:
            raise ValueError("resume_range start must not exceed stop")
        record = self._checkpoint_record
        resume_step = start
        if record is not None:
            step = record.get("step")
            if step is not None:
                resume_step = max(start, _nonnegative_integer(step, "checkpoint step"))
        if resume_step >= stop:
            return iter(())
        return iter(range(resume_step, stop))

    def checkpoint(self, position: int, state: Any) -> None:
        """Persist an explicit checkpoint at ``position`` with the given ``state``.

        ``state`` may be any value; non-dictionaries are wrapped internally so
        the storage format stays uniform.  The user owns the loop and decides
        what to save and when.
        """
        if self._checkpoint is None:
            raise RuntimeError("checkpoint() requires a managed Stage context")
        position = _nonnegative_integer(position, "checkpoint position")
        stored_state = state if isinstance(state, dict) else {"__payload__": state}
        self._checkpoint.save(step=position, state=stored_state)
        # Keep the in-memory record in sync so the completed snapshot can use
        # the latest explicit state without re-reading disk.
        record = dict(self._checkpoint_record) if self._checkpoint_record else {}
        record["step"] = position
        record["state"] = stored_state
        object.__setattr__(self, "_checkpoint_record", record)
        object.__setattr__(self, "_explicit_checkpoint_used", True)
