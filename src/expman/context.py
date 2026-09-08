"""Run-scoped services and nested execution observation."""

import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from numbers import Real
from time import perf_counter
from typing import Literal
from uuid import uuid4

from .events import Event, ExecutionEvent, MetricEvent, ProgressEvent, Status
from .recorders import InMemoryRecorder, Recorder

_logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _nonnegative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _name(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True)
class RunContext:
    """Shared recorder and identity, with an immutable execution scope.

    Each call to observe creates a child scope while retaining the run ID and
    recorder. Ordinary recorder exceptions are logged and suppressed so an
    unavailable monitoring sink does not change the business outcome.
    """

    run_id: str = field(default_factory=lambda: uuid4().hex)
    recorder: Recorder = field(default_factory=InMemoryRecorder)
    _execution_id: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _name(self.run_id, "run_id")

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
        self, name: str, *, kind: Literal["pipeline", "stage"]
    ) -> Iterator["RunContext"]:
        """Record start and finish, preserving the original business exception."""
        _name(name, "execution name")
        if kind not in ("pipeline", "stage"):
            raise ValueError("kind must be 'pipeline' or 'stage'")
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
                Status.CANCELLED
                if isinstance(error, (KeyboardInterrupt, SystemExit))
                else Status.FAILED
            )
            error_type = type(error).__qualname__
            try:
                error_message = str(error)
            except Exception:
                # A user-defined exception may itself fail to render.
                error_message = "<exception message unavailable>"
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
            )
        )
