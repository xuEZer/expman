"""Per-worker cgroup memory limits, and the signal that one was hit.

One runaway experiment must not be able to consume the host, so each worker runs
inside a cgroup whose ``memory.max`` is derived from that attempt's estimated peak.
Direct cgroupfs is used when this process may write there; otherwise a transient
systemd user scope carries the same limits, because systemd can create the cgroup
even where the mount is read-only for us.

A worker killed by its own cap would take the evidence with it, so a watcher inside
the worker writes the cap record as soon as the kernel reports that the limit
refused a charge (``memory.events`` ``max``/``oom_kill``), and the parent also
samples the cgroup while the worker lives.
"""

import shutil
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from . import devices
from .storage import PickleSerializer, write_record

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


def own_cgroup() -> Path | None:
    """This process's own cgroup directory, or None when it cannot be read."""
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("0::"):
            return CGROUP_ROOT / line.split("::", 1)[1].strip().lstrip("/")
    return None


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
    parent = own_cgroup()
    cgroup = None if parent is None else parent.parent / f"{unit}.scope"
    return MemoryLimit(prefix, cgroup, limit_kb, "systemd")


def memory_limit(unit: str, estimate_kb: float) -> MemoryLimit:
    """Cheapest mechanism that can cap one worker; uncapped when none is available."""
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


def limit_of(directory: Path | None) -> float:
    """The cgroup's own memory.max in kilobytes, or 0.0 when unreadable or unset."""
    if directory is None:
        return 0.0
    try:
        value = (directory / "memory.max").read_text().strip()
    except OSError:
        return 0.0
    return 0.0 if value == "max" else float(value.split()[0]) / 1024


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

    The counters only move once the kernel is out of options, so a worker also
    reports a peak that came close to its own limit (see ``near_cap``): that record
    is written before the kill, while a kill-path counter may never be observed.
    """
    return any(
        counts.get(key, 0.0) > 0
        for key in ("high", "max", "oom", "oom_kill", "oom_group_kill")
    )


def near_cap(peak_kb: float, limit_kb: float) -> bool:
    """Whether an observed peak came close enough to its limit to have hit it."""
    return limit_kb > 0 and peak_kb >= limit_kb * devices.CAPACITY_NEAR


def watch(record: Path, interval: float, stop) -> None:
    """Write this worker's cap record as soon as its limit refuses a charge.

    Runs inside the worker: a cgroup OOM kill would otherwise leave the parent with
    nothing but an unexplained SIGKILL. The record is written before the kill, and
    written again when the watcher stops, so a normal exit still reports the peak
    (the event counters are cumulative, so the later write keeps the evidence).
    """
    directory = own_cgroup()
    limit_kb = limit_of(directory)
    while not stop.wait(interval):
        counts = events(directory)
        observed = usage(directory)
        capped = observed is not None and near_cap(observed[1], limit_kb)
        if refused(counts) or capped:
            write_record(
                record, {"events": counts, "usage": observed}, PickleSerializer()
            )
            break
    write_record(
        record,
        {"events": events(directory), "usage": usage(directory)},
        PickleSerializer(),
    )
