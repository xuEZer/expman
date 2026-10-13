"""Data objects and tiny helpers used by the scheduler."""

import os
import signal
import subprocess
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from expman.core.experiment import StageResult
from expman.estimation.duration_model import StageDuration
from expman.scheduling.ipc import Channel


class HostGate:
    """Admission from the current host-memory observation only."""

    def can_launch(self, host):
        return not (host is None or host.tight)


@dataclass
class Worker:
    process: subprocess.Popen
    experiment: object
    device: int | None
    uuid: str | None
    root: Path
    started: float
    stage_index: int | None = 0
    stage_attempt: int = 1
    channel: Channel | None = None
    # The peak estimates admission charges this attempt for. Admission reserves
    # these rather than the hard ceilings below, so a packed plan spends what
    # attempts are expected to need and the ceilings stay an allowance on top.
    host_reserve_kb: float = 0.0
    gpu_reserve_kb: float = 0.0
    # The device ceiling this attempt actually runs under, or zero when it was
    # started without one. Admission charges only the growth above what it has
    # already reserved, so an attempt with no ceiling charges its own usage.
    gpu_cap_kb: float = 0.0
    expected: StageDuration | None = None
    area: float = 0.0
    limit: object = None
    # Sampled every tick: resident is what the worker holds now, peak is the most
    # it has ever held. Admission reserves the gap between the two.
    resident_kb: float = 0.0
    peak_kb: float = 0.0
    gpu_allocated_kb: float | None = None
    gpu_reserved_kb: float | None = None
    gpu_peak_allocated_kb: float | None = None
    gpu_peak_reserved_kb: float | None = None
    memory_events: dict = field(default_factory=dict)
    finished: StageResult | None = None
    dependencies: list | None = None
    reused: bool = False
    cap_hit: bool = False


@dataclass(frozen=True)
class Candidate:
    """One ready Stage with its resource reservations for this scheduling tick."""

    run_id: str
    stage_index: int
    host_reserve_kb: float
    gpu_reserve_kb: float
    gpu_cap_kb: float
    expected: StageDuration | None
    novelty: float


@dataclass(frozen=True)
class Plan:
    """A feasible placement of ready Stage candidates on the selected cards."""

    host_left_kb: float
    gpu_left_kb: tuple[float, ...]
    placements: tuple[tuple[Candidate, int], ...] = ()
    novelty: float = 0.0


def _kill_group(process):
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)


def _close_stdin(process):
    """Close a worker's control pipe; workers always start with stdin=PIPE."""
    if process.stdin is not None:
        process.stdin.close()
