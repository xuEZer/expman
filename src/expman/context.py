"""Run-scoped services and nested execution observation."""

import logging
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from numbers import Real
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from .events import Event, ExecutionEvent, MetricEvent, ProgressEvent, Status
from .frozen import freeze
from .recorders import InMemoryRecorder, Recorder
from .storage import Checkpoint, RunStore

_logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _nonnegative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


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
    state: dict[str, Any] = field(default_factory=dict, kw_only=True)
    stage_id: int | None = field(default=None, kw_only=True)
    _store: RunStore | None = field(default=None, kw_only=True, repr=False)
    _stage_path: tuple[int, ...] = field(default=(), kw_only=True, repr=False)
    _checkpoint: Checkpoint | None = field(default=None, kw_only=True, repr=False)
    _pipeline_calls: dict = field(default_factory=dict, kw_only=True, repr=False)
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
        if not isinstance(self.state, dict):
            raise TypeError("state must be a dictionary")

    @property
    def checkpoint(self) -> Checkpoint:
        if self._checkpoint is None:
            raise RuntimeError(
                "checkpoint requires a stage managed by Experiment or Batch"
            )
        return self._checkpoint

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
