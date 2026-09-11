"""Device selection plus whole-device and host memory observations."""

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import ConfigError

# Free-ratio floor for device memory. At 1% a launch only stops and shedding only
# starts once a card is nearly exhausted: about 82 MiB of an 8 GiB card, so the OOM
# margin is left to the user.
MEMORY_MARGIN = 0.01
# Host RAM kept free of experiment allocations, in kilobytes. Host memory is shared
# by every worker process, and an exhausted host does not fail the allocating
# process the way a full card fails a kernel launch: the kernel reclaims, swaps, or
# kills instead. A bursty allocator needs an absolute reserve, so this is fixed
# rather than a fraction of total RAM: a fraction grows with the machine and keeps
# admitting launches on a large host well past the point where one more multi-GB
# allocation, or a few simultaneous ones, stalls every process in the VM. The
# reserve also keeps a working set of page cache alive: once every page of a data
# set has been evicted, each read goes back to the disk and the host looks loaded.
HOST_RESERVE_KB = 2 * 1024 * 1024
# Assumed peak resident memory of an attempt that has never been observed, in
# kilobytes. Admission reserves the sum of expected peaks, so this bounds what one
# unknown attempt may hold; over-estimating costs throughput, under-estimating
# costs the host.
HOST_PEAK_KB_DEFAULT = 1024 * 1024
# One scheduler tick: how often memory is re-read, finished attempts are collected,
# blocks are released and at most one attempt per card is launched. A slower tick
# launches more gently and spends less time inside nvidia-smi, but it also delays
# collecting a finished attempt and shedding under pressure by up to one interval.
POLL_INTERVAL = 1.0
# Measured on RTX 3070 laptop / WSL2, one GPU, at the scheduler's own cadence:
# median 53 ms, p99 300 ms, worst 620 ms with the GPU in desktop use; median
# 143 ms, worst 915 ms with the host CPU saturated. A false timeout kills a live
# attempt and blocks refills, so keep a few times the worst observation.
QUERY_TIMEOUT = 3.0
MEMINFO_PATH = Path("/proc/meminfo")
VMSTAT_PATH = Path("/proc/vmstat")
PROC_ROOT = Path("/proc")
# WSL2 installs the Linux nvidia-smi here without adding it to PATH.
NVIDIA_SMI_FALLBACKS = (Path("/usr/lib/wsl/lib/nvidia-smi"),)


class MemoryObservationError(RuntimeError):
    """A failed or timed-out memory query; the scheduler treats host RAM as tight."""


def configured_devices(configs) -> tuple[int, ...]:
    selections = []
    for cfg in configs:
        value = cfg.get("device", [])
        if (
            not isinstance(value, list)
            or any(type(item) is not int or item < 0 for item in value)
            or len(set(value)) != len(value)
        ):
            raise ConfigError(
                "device must be a list of distinct nonnegative GPU indices"
            )
        selections.append(tuple(value))
    if selections and any(item != selections[0] for item in selections):
        raise ConfigError("device must be the same for every experiment in a Batch")
    return selections[0] if selections else ()


@dataclass(frozen=True)
class DeviceMemory:
    uuid: str
    total: float
    free: float

    @property
    def free_ratio(self) -> float:
        return self.free / self.total


def nvidia_smi() -> str:
    """Resolve nvidia-smi from PATH, then from the known WSL2 driver directory."""
    found = shutil.which("nvidia-smi")
    if found is not None:
        return found
    for candidate in NVIDIA_SMI_FALLBACKS:
        if candidate.is_file():
            return str(candidate)
    return "nvidia-smi"


class NvidiaMemory:
    def __init__(self, devices):
        self.devices = devices
        self.identities = None

    def sample(self) -> dict[int, DeviceMemory]:
        try:
            result = subprocess.run(
                [
                    nvidia_smi(),
                    "--query-gpu=index,uuid,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                    "--id=" + ",".join(map(str, self.devices)),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=QUERY_TIMEOUT,
            )
            values = {}
            for line in result.stdout.splitlines():
                index, uuid, total, free = [item.strip() for item in line.split(",")]
                total, free = float(total), float(free)
                if (
                    not uuid.startswith("GPU-")
                    or not math.isfinite(total)
                    or not math.isfinite(free)
                    or total <= 0
                    or not 0 <= free <= total
                ):
                    raise ValueError("invalid GPU memory observation")
                values[int(index)] = DeviceMemory(uuid, total, free)
            if set(values) != set(self.devices):
                raise ValueError("selected GPUs are missing from nvidia-smi")
            identities = {index: value.uuid for index, value in values.items()}
            if self.identities is not None and identities != self.identities:
                raise RuntimeError("GPU identities changed while the Batch was running")
            self.identities = identities
            return values
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            raise MemoryObservationError(
                f"could not query selected NVIDIA GPUs: {error}"
            ) from error


@dataclass(frozen=True)
class HostMemory:
    """Host RAM reported by /proc/meminfo; either reading can stop launches."""

    total_kb: float
    available_kb: float
    swap_total_kb: float
    swap_free_kb: float
    # The kernel paged at least once since the previous reading. Occupancy is not
    # this signal: swapped pages stay out until something touches them, so a few
    # stale megabytes outlive the shortage that pushed them out and, as a gate,
    # would hold every launch back for the rest of the session.
    paging: bool = False

    @property
    def available_ratio(self) -> float:
        return self.available_kb / self.total_kb

    @property
    def headroom_kb(self) -> float:
        """MemAvailable above the reserve; a non-positive value admits nothing."""
        return self.available_kb - HOST_RESERVE_KB

    @property
    def tight(self) -> bool:
        """Whether host RAM cannot take another launch without risk."""
        return self.headroom_kb <= 0 or self.paging


def fits_reserve(host: HostMemory | None, needed_kb: float) -> bool:
    """Whether one more allocation of needed_kb fits above the host reserve.

    Admission reserves the sum of expected peaks instead of trusting a snapshot:
    spending memory that attempts already in flight are still going to ask for is
    what fills the host while every reading along the way looked healthy.
    """
    return host is not None and not host.tight and host.headroom_kb >= needed_kb


class MeminfoMonitor:
    """Read host RAM availability for scheduling; a failed read pauses launches."""

    def __init__(self, path: Path | None = None, vmstat_path: Path | None = None):
        self.path = MEMINFO_PATH if path is None else Path(path)
        self.vmstat_path = VMSTAT_PATH if vmstat_path is None else Path(vmstat_path)
        self._paged: float | None = None

    def sample(self) -> HostMemory:
        try:
            fields = {}
            for line in self.path.read_text().splitlines():
                key, _, value = line.partition(":")
                if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                    fields[key] = float(value.split()[0])
            total = fields["MemTotal"]
            available = fields["MemAvailable"]
            swap_total = fields.get("SwapTotal", 0.0)
            swap_free = fields.get("SwapFree", 0.0)
        except (OSError, ValueError, IndexError, KeyError) as error:
            raise MemoryObservationError(
                f"could not observe host memory: {error}"
            ) from error
        if (
            not math.isfinite(total)
            or total <= 0
            or not math.isfinite(available)
            or not 0 <= available <= total
            or not math.isfinite(swap_total)
            or swap_total < 0
            or not math.isfinite(swap_free)
            or not 0 <= swap_free <= swap_total
        ):
            raise MemoryObservationError("invalid host memory observation")
        return HostMemory(total, available, swap_total, swap_free, self._paging())

    def _paging(self) -> bool:
        """Whether the kernel paged in the interval since the previous sample.

        The paging counters only move while the kernel is short of the RAM it
        wanted, and a delta cannot latch the way swap occupancy does. A missing
        or unreadable /proc/vmstat is an extra trip rather than the primary
        reading, so it reports no paging instead of pausing launches.
        """
        paged = 0.0
        try:
            for line in self.vmstat_path.read_text().splitlines():
                key, _, value = line.partition(" ")
                if key in ("pswpin", "pswpout"):
                    paged += float(value.split()[0])
        except (OSError, ValueError, IndexError):
            self._paged = None
            return False
        previous, self._paged = self._paged, paged
        return previous is not None and paged > previous


def residency_kb(pid: int) -> tuple[float, float] | None:
    """Current and peak resident set of a live process in kB; None if unreadable.

    Peak is what admission charges a running worker for: the gap between what it
    holds now and the most it has been seen to need.
    """
    try:
        current = peak = None
        for line in (PROC_ROOT / str(pid) / "status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                current = float(line.split()[1])
            elif line.startswith("VmHWM:"):
                peak = float(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    if current is None or peak is None:
        return None
    return current, peak
