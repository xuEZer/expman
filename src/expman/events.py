"""Immutable events emitted by executions and user-defined stages."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal, TypeAlias


class Status(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ExecutionEvent:
    run_id: str
    execution_id: str
    parent_id: str | None
    kind: Literal["pipeline", "stage"]
    name: str
    status: Status
    timestamp: datetime
    duration_seconds: float | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class MetricEvent:
    run_id: str
    execution_id: str | None
    name: str
    value: float
    step: int | None
    timestamp: datetime


@dataclass(frozen=True)
class ProgressEvent:
    run_id: str
    execution_id: str | None
    completed: int
    total: int | None
    unit: str
    timestamp: datetime


Event: TypeAlias = ExecutionEvent | MetricEvent | ProgressEvent
