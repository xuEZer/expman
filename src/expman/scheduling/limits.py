"""Per-attempt memory accounting and verified cgroup v1/v2 limits.

Mount locations come from proc, including hybrid hierarchies. A systemd property
request is not proof that the memory controller enforces it; workers verify the
actual independent group's counters and limit after entering their scope.
"""

import math
import re
import shutil
import subprocess
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import expman.config.devices as devices

PROC_ROOT = Path("/proc")
SYSTEMD_RUN = "systemd-run"
EVENT_KEYS = ("high", "max", "oom", "oom_kill", "oom_group_kill")
# v1 encodes its unlimited sentinel as a page-aligned LONG_MAX.
_V1_UNLIMITED_BYTES = 1 << 60
_OBSERVED_PEAKS: dict[Path, float] = {}


@dataclass(frozen=True)
class MemoryLimit:
    """Requested launch contract; capped only after kernel verification."""

    command: tuple[str, ...]
    cgroup: Path | None
    cap_kb: float
    mechanism: str
    verified: bool = False

    @property
    def capped(self) -> bool:
        return self.cap_kb > 0 and self.verified


def cap_kb(estimate_kb) -> float:
    """Cgroup limit for one worker: estimate plus room, never below the floor."""
    return max(devices.CAPACITY_FLOOR_KB, estimate_kb * devices.CAPACITY_FACTOR)


def _unescape(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def _cgroup_directory(lines, mount_lines) -> Path | None:
    """Resolve the memory controller's membership through its real mount root."""
    memberships = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, membership = parts
        if "memory" in controllers.split(","):
            memberships.insert(0, ("cgroup", membership))
        elif hierarchy == "0" and not controllers:
            memberships.append(("cgroup2", membership))
    mounts = []
    for line in mount_lines:
        before, separator, after = line.partition(" - ")
        fields, filesystem = before.split(), after.split()
        if not separator or len(fields) < 5 or len(filesystem) < 3:
            continue
        kind = filesystem[0]
        if kind == "cgroup" and "memory" not in filesystem[2].split(","):
            continue
        if kind in ("cgroup", "cgroup2"):
            mounts.append((kind, _unescape(fields[3]), _unescape(fields[4])))
    # A hybrid machine's v1 memory membership takes precedence over 0::, which
    # can describe a unified systemd scope with no memory controller at all.
    for kind, membership in memberships:
        for filesystem, root, mountpoint in mounts:
            if filesystem != kind:
                continue
            try:
                relative = PurePosixPath(_unescape(membership)).relative_to(root)
            except ValueError:
                continue
            if ".." not in relative.parts:
                return Path(mountpoint).joinpath(*relative.parts)
    return None


def _memory_cgroup(pid: int | str) -> Path | None:
    try:
        root = PROC_ROOT / str(pid)
        memberships = (root / "cgroup").read_text().splitlines()
        mounts = (root / "mountinfo").read_text().splitlines()
    except OSError:
        return None
    return _cgroup_directory(memberships, mounts)


def own_cgroup() -> Path | None:
    """The current process's actual memory-controller group, possibly shared."""
    return _memory_cgroup("self")


def cgroup_for_pid(pid: int) -> Path | None:
    """Resolve a live process's memory controller, including hybrid/v1 mounts."""
    return _memory_cgroup(pid)


def attempt_cgroup(directory: Path | None, unit: str | None) -> Path | None:
    """Only an exact attempt unit (or its descendant) is independent accounting."""
    if directory is None or not unit or not unit.startswith("expman-"):
        return None
    for candidate in (directory, *directory.parents):
        if candidate.name in (unit, f"{unit}.scope"):
            return candidate
    return None


def _read_number(path: Path) -> float | None:
    try:
        value = float(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _is_v1(directory: Path | None) -> bool:
    return directory is not None and (directory / "memory.usage_in_bytes").is_file()


def memory_cap_kb(directory: Path | None) -> float | None:
    """Kernel limit in KiB; 0 means unlimited, None means it cannot be read."""
    if directory is None:
        return None
    filename = "memory.limit_in_bytes" if _is_v1(directory) else "memory.max"
    try:
        raw = (directory / filename).read_text().strip()
    except OSError:
        return None
    if raw == "max":
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        return None
    if not math.isfinite(value) or value < 0:
        return None
    if filename == "memory.limit_in_bytes" and value >= _V1_UNLIMITED_BYTES:
        return 0.0
    return value / 1024


def cap_matches_request(actual_kb: float | None, requested_kb: float) -> bool:
    """A finite kernel cap no looser than the request, allowing page rounding."""
    return (
        actual_kb is not None
        and math.isfinite(actual_kb)
        and actual_kb > 0
        and math.isfinite(requested_kb)
        and requested_kb > 0
        and actual_kb <= requested_kb + 4.0
    )


def _direct(unit: str, limit_kb: float) -> Path | None:
    """Create an independent memory group when this hierarchy is writable."""
    parent = own_cgroup()
    if parent is None or not parent.is_dir():
        return None
    v1 = _is_v1(parent)
    directory = parent.parent / unit
    created = False
    try:
        directory.mkdir()
        created = True
        if v1:
            (directory / "memory.limit_in_bytes").write_text(str(int(limit_kb * 1024)))
            swap = directory / "memory.memsw.limit_in_bytes"
            if swap.is_file():
                swap.write_text(str(int(limit_kb * 1024)))
        else:
            (directory / "memory.max").write_text(str(int(limit_kb * 1024)))
            (directory / "memory.swap.max").write_text("0")
    except OSError:
        if created:
            with suppress(OSError):
                directory.rmdir()
        return None
    return directory


def _scope(unit: str, limit_kb: float) -> MemoryLimit | None:
    """Request systemd enforcement; the worker must verify actual placement."""
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
    properties = (
        (f"--property=MemoryLimit={int(limit_kb)}K",)
        if _is_v1(own_cgroup())
        else (f"--property=MemoryMax={int(limit_kb)}K", "--property=MemorySwapMax=0")
    )
    prefix = (
        SYSTEMD_RUN,
        "--user",
        "--scope",
        "--quiet",
        f"--unit={unit}",
        *properties,
        "--property=MemoryAccounting=yes",
        "--property=KillMode=control-group",
        "--",
    )
    return MemoryLimit(prefix, None, limit_kb, "systemd")


def memory_limit(unit: str, estimate_kb: float) -> MemoryLimit:
    """Create a direct cap or request a scope; never claim unverified protection."""
    requested = cap_kb(estimate_kb)
    directory = _direct(unit, requested)
    if directory is not None:
        verified = (
            attempt_cgroup(directory, unit) is not None
            and usage(directory) is not None
            and cap_matches_request(memory_cap_kb(directory), requested)
        )
        return MemoryLimit((), directory, requested, "cgroup", verified=verified)
    scope = _scope(unit, requested)
    if scope is not None:
        return scope
    return MemoryLimit((), None, 0.0, "none")


def join(directory: Path, pid: int) -> bool:
    """Move a started child into a directly created v1 or v2 cgroup."""
    try:
        (directory / "cgroup.procs").write_text(str(pid))
    except OSError:
        return False
    return True


def usage(directory: Path | None) -> tuple[float, float] | None:
    """Current/peak KiB; missing native peak retains the observed high water."""
    if directory is None:
        return None
    current_name, peak_name = (
        ("memory.usage_in_bytes", "memory.max_usage_in_bytes")
        if _is_v1(directory)
        else ("memory.current", "memory.peak")
    )
    current = _read_number(directory / current_name)
    if current is None:
        return None
    native_peak = _read_number(directory / peak_name)
    peak = max(current, native_peak or 0.0, _OBSERVED_PEAKS.get(directory, 0.0))
    _OBSERVED_PEAKS[directory] = peak
    return current / 1024, peak / 1024


def events(directory: Path | None) -> dict[str, float]:
    """Read this memory group's refusal counters on v1 or v2."""
    if directory is None:
        return {}
    if _is_v1(directory):
        failed = _read_number(directory / "memory.failcnt")
        return {} if failed is None else {"max": failed}
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
    """Whether this independent group's memory controller refused a charge."""
    return any(counts.get(key, 0.0) > 0 for key in EVENT_KEYS)


def near_cap(peak_kb: float, limit_kb: float) -> bool:
    """Whether an observed peak came close enough to its limit to have hit it."""
    return limit_kb > 0 and peak_kb >= limit_kb * devices.CAPACITY_NEAR
