"""Parent-side state and IPC handling for one Stage worker process."""

import math
import os
import signal
import subprocess
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TypeGuard

from expman.core.events import ExecutionEvent, ProgressEvent, Status
from expman.core.experiment import Experiment, StageResult
from expman.core.recorders import Recorder, record_event
from expman.estimation.duration_model import StageDuration
from expman.resources.limits import MemoryLimit, cap_matches_request
from expman.scheduling.ipc import Channel


def _memory_value(value) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


@dataclass
class Worker:
    process: subprocess.Popen
    experiment: Experiment
    device: int | None
    uuid: str | None
    started: float
    stage_index: int = 0
    stage_attempt: int = 1
    channel: Channel | None = None
    # The peak estimates admission charges this attempt for. Admission reserves
    # these rather than the hard ceilings below, so a packed plan spends what
    # attempts are expected to need and the ceilings stay an allowance on top.
    host_reserve_kb: float = 0.0
    gpu_reserve_kb: float = 0.0
    # Actual MiB-granularity request sent in the MPS client environment.
    gpu_cap_kb: float = 0.0
    mps_client: tuple[int, int] | None = None
    mps_slots_full: bool = False

    expected: StageDuration | None = None
    area: float = 0.0
    limit: MemoryLimit | None = None
    # Sampled every tick: resident is what the worker holds now, peak is the most
    # it has ever held. Admission reserves the gap between the two.
    resident_kb: float = 0.0
    peak_kb: float = 0.0
    host_usage_source: str | None = None
    host_usage_error: str | None = None
    host_limit_kb: float | None = None
    gpu_used_kb: float | None = None
    gpu_process_peak_kb: float | None = None
    memory_events: dict = field(default_factory=dict)
    finished: StageResult | None = None
    progress: ProgressEvent | None = None
    _stage_execution_id: str | None = None
    _stage_progress: ProgressEvent | None = None
    # Time of the last usable top-level report and seconds left at that report.
    progress_forecast: tuple[float, float] | None = None
    dependencies: list | None = None
    reused: bool = False
    cap_hit: bool = False

    @property
    def gpu_limit_verified(self) -> bool:
        return self.mps_client is not None

    def read_messages(self, recorder: Recorder) -> None:
        """Consume worker IPC without querying its process or cgroup."""
        if self.channel is None:
            return
        for message in self.channel.drain():
            if not isinstance(message, dict):
                continue
            kind = message.get("type")
            if kind == "event":
                event = message["event"]
                if isinstance(event, (ExecutionEvent, ProgressEvent)):
                    self._observe_progress(event)
                record_event(recorder, event)
            elif kind == "mps_ready":
                self.mps_client = (message["server"], message["pid"])
            elif kind == "mps_slots_full":
                self.mps_slots_full = True
            elif kind == "startup_failed":
                self.finished = StageResult(
                    self.experiment.run_id,
                    self.stage_index,
                    self.stage_attempt,
                    Status.FAILED,
                    0.0,
                    message["error_type"],
                    message["error_message"],
                )
            elif kind == "resource":
                self.update_resources(message)
            elif kind == "finished" and isinstance(message.get("result"), StageResult):
                self.finished = message["result"]
                dependencies = message.get("dependencies")
                self.dependencies = (
                    dependencies if isinstance(dependencies, list) else None
                )
                self.reused = message.get("reused") is True

    def _observe_progress(self, event: ExecutionEvent | ProgressEvent) -> None:
        if (
            event.run_id != self.experiment.run_id
            or event.attempt != self.stage_attempt
        ):
            return
        if isinstance(event, ExecutionEvent):
            # The scheduled Stage starts before any nested Stage with the same index.
            if (
                self._stage_execution_id is None
                and event.kind == "stage"
                and event.status == Status.RUNNING
                and event.stage_id == self.stage_index
                and not event.reused
            ):
                self._stage_execution_id = event.execution_id
            return
        self.progress = event
        if (
            self._stage_execution_id is None
            or event.execution_id != self._stage_execution_id
        ):
            return
        previous = self._stage_progress
        forecast = None
        if (
            previous is not None
            and event.total == previous.total
            and event.unit == previous.unit
        ):
            seconds = (event.timestamp - previous.timestamp).total_seconds()
            if seconds < 0 or event.completed == previous.completed:
                return  # A heartbeat contributes neither new work nor a new baseline.
            if (
                seconds > 0
                and event.total is not None
                and previous.completed < event.completed < event.total
            ):
                with suppress(OverflowError):
                    units_left = event.total - event.completed
                    units_done = event.completed - previous.completed
                    seconds_left = seconds * (units_left / units_done)
                    if math.isfinite(seconds_left) and seconds_left > 0:
                        forecast = (event.timestamp.timestamp(), seconds_left)
        self._stage_progress = event
        self.progress_forecast = forecast

    def update_resources(self, message) -> None:
        current, peak = message.get("host_current_kb"), message.get("host_peak_kb")
        if _memory_value(current) and _memory_value(peak):
            self.resident_kb = current
            self.peak_kb = max(self.peak_kb, current, peak)
        for name in ("host_usage_source", "host_usage_error"):
            value = message.get(name)
            if value is None or isinstance(value, str):
                setattr(self, name, value)
        if "host_limit_verified" in message and self.limit is not None:
            observed_cap = message.get("host_limit_kb")
            valid_cap = _memory_value(observed_cap)
            self.host_limit_kb = observed_cap if valid_cap else None
            verified = (
                message["host_limit_verified"] is True
                and valid_cap
                and cap_matches_request(observed_cap, self.limit.cap_kb)
            )
            # cap_kb retains the requested ceiling, even if this observation
            # cannot verify an effective per-attempt kernel limit.
            self.limit = replace(self.limit, verified=verified)
        events = message.get("memory_events")
        if isinstance(events, dict):
            self.memory_events = events
        directory = message.get("cgroup")
        if self.limit is not None and "cgroup" in message:
            self.limit = replace(
                self.limit,
                cgroup=Path(directory) if isinstance(directory, str) else None,
            )


def _kill_group(process):
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)


def _close_stdin(process):
    """Close a worker's control pipe; workers always start with stdin=PIPE."""
    if process.stdin is not None:
        process.stdin.close()
