"""Per-worker cgroup memory limits, and the signal that one was hit.

One runaway experiment must not be able to consume the host, so each worker runs
inside a cgroup whose ``memory.max`` is derived from that attempt's estimated peak.
Direct cgroupfs is used when this process may write there; otherwise a transient
systemd user scope carries the same limits, because systemd can create the cgroup
even where the mount is read-only for us.

A Stage whose first attempt has not run yet has no measured peak to cap against,
so that attempt runs uncapped and the peak it reaches becomes the sample that
later attempts are capped from.

The worker reports its cgroup counters through the scheduler IPC channel on each
tick.  If the kernel kills it before that final report, the parent reads only the
worker's cgroup counters once during exit handling; no per-tick parent-side cgroup
polling is needed.
"""

import shutil
import subprocess
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from . import devices

CGROUP_ROOT = Path("/sys/fs/cgroup")
SYSTEMD_RUN = "systemd-run"
EVENT_KEYS = ("high", "max", "oom", "oom_kill", "oom_group_kill")


@dataclass(frozen=True)
class MemoryLimit:
    """How to start one capped worker, and where to read what it used."""

    command: tuple[str, ...]
    cgroup: Path | None
    cap_kb: float
    mechanism: str

    @property
    def capped(self) -> bool:
        return self.cap_kb > 0


def cap_kb(estimate_kb) -> float:
    """Cgroup limit for one worker: the estimate plus room, never below the floor."""
    return max(devices.CAPACITY_FLOOR_KB, estimate_kb * devices.CAPACITY_FACTOR)


def _cgroup_directory(lines) -> Path | None:
    """Resolve the unified-v2 entry from a ``/proc/*/cgroup`` file."""
    for line in lines:
        if line.startswith("0::"):
            return CGROUP_ROOT / line.split("::", 1)[1].strip().lstrip("/")
    return None


def own_cgroup() -> Path | None:
    """This process's own cgroup directory, or None when it cannot be read."""
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    return _cgroup_directory(lines)


def cgroup_for_pid(pid: int) -> Path | None:
    """Return a live process's cgroup directory without guessing its hierarchy."""
    try:
        lines = (Path("/proc") / str(pid) / "cgroup").read_text().splitlines()
    except OSError:
        return None
    return _cgroup_directory(lines)


def _direct(unit: str, limit_kb: float) -> Path | None:
    """Create this run's own cgroup; None when the mount does not allow it."""
    parent = own_cgroup()
    if parent is None or not parent.is_dir():
        return None
    directory = parent.parent / unit
    try:
        directory.mkdir()
        (directory / "memory.max").write_text(str(int(limit_kb * 1024)))
        (directory / "memory.swap.max").write_text("0")
    except OSError:
        with suppress(OSError):
            directory.rmdir()
        return None
    return directory


def _scope(unit: str, limit_kb: float) -> MemoryLimit | None:
    """Transient systemd user scope carrying the same limits."""
    if shutil.which(SYSTEMD_RUN) is None:
        return None
    try:
        subprocess.run(
            ("systemctl", "--user", "show-environment"),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    prefix = (
        SYSTEMD_RUN,
        "--user",
        "--scope",
        "--quiet",
        f"--unit={unit}",
        f"--property=MemoryMax={int(limit_kb)}K",
        "--property=MemorySwapMax=0",
        "--property=MemoryAccounting=yes",
        "--property=KillMode=control-group",
        "--",
    )
    # A user manager owns the scope's hierarchy, which need not resemble the
    # scheduler's own cgroup.  The worker discovers its actual cgroup after
    # systemd has placed it there; guessing from this process produces a path
    # that exists nowhere and silently turns its peak reports into zeroes.
    return MemoryLimit(prefix, None, limit_kb, "systemd")


def memory_limit(
    unit: str, estimate_kb: float, *, cold_start: bool = False
) -> MemoryLimit:
    """Cheapest mechanism that can cap one worker; uncapped when none is available.

    ``cold_start`` marks a Stage that has yet to complete an attempt in this
    Batch. Nothing has measured what that Stage needs, so the only estimate
    available is the framework's own default and capping from it would refuse an
    attempt whose true peak is still unknown. The first attempt therefore runs
    uncapped and teaches the estimator, which caps every attempt after it.

    The short circuit must precede ``_direct``/``_scope``: a zero limit reaching
    ``_direct`` would be written as ``memory.max = 0``, which in cgroup v2 forbids
    the worker any memory at all.
    """
    if cold_start:
        return MemoryLimit((), None, 0.0, "cold-start")
    limit = cap_kb(estimate_kb)
    directory = _direct(unit, limit)
    if directory is not None:
        return MemoryLimit((), directory, limit, "cgroup")
    scope = _scope(unit, limit)
    if scope is not None:
        return scope
    return MemoryLimit((), None, 0.0, "none")


def join(directory: Path, pid: int) -> bool:
    """Move a started child into a directly created cgroup."""
    try:
        (directory / "cgroup.procs").write_text(str(pid))
    except OSError:
        return False
    return True


def usage(directory: Path | None) -> tuple[float, float] | None:
    """Current and peak resident kilobytes of a cgroup, or None if unreadable."""
    if directory is None:
        return None
    try:
        current = float((directory / "memory.current").read_text().split()[0])
        peak = float((directory / "memory.peak").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return current / 1024, peak / 1024


def events(directory: Path | None) -> dict[str, float]:
    """The cgroup's memory event counters; empty when unreadable."""
    if directory is None:
        return {}
    counts = {}
    try:
        for line in (directory / "memory.events").read_text().splitlines():
            key, _, value = line.partition(" ")
            if key in EVENT_KEYS:
                counts[key] = float(value.split()[0])
    except (OSError, ValueError, IndexError):
        return {}
    return counts


def refused(counts: dict[str, float]) -> bool:
    """Whether the kernel reports that this cgroup's limit refused a charge.

    The counters only move once the kernel is out of options.  The worker also
    reports a peak that came close to its own limit (see ``near_cap``), which
    catches a kill path before the parent can receive a final IPC report.
    """
    return any(
        counts.get(key, 0.0) > 0
        for key in ("high", "max", "oom", "oom_kill", "oom_group_kill")
    )


def near_cap(peak_kb: float, limit_kb: float) -> bool:
    """Whether an observed peak came close enough to its limit to have hit it."""
    return limit_kb > 0 and peak_kb >= limit_kb * devices.CAPACITY_NEAR
