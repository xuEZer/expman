"""GPU and host resource observations and scheduler memory budgets."""

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from expman.settings import settings

# Device memory carries no ratio threshold: CUDA OOM is treated as evidence that
# the Stage needs a larger next contract. Signatures come from PyTorch's own
# error type and message.
CUDA_OOM_TYPE = "OutOfMemoryError"
CUDA_OOM_MARKERS = (
    "CUDA out of memory",
    "CUDA error: out of memory",
    "cuDNN error: CUDNN_STATUS_ALLOC_FAILED",
)
MEMINFO_PATH = Path("/proc/meminfo")
# WSL2 installs the Linux nvidia-smi here without adding it to PATH.
NVIDIA_SMI_FALLBACKS = (Path("/usr/lib/wsl/lib/nvidia-smi"),)


class MemoryObservationError(RuntimeError):
    """A failed or timed-out memory query; the scheduler treats host RAM as tight."""


def cuda_out_of_memory(error_type, error_message) -> bool:
    """Whether a failed attempt exceeded the CUDA memory available to it."""
    if error_type == CUDA_OOM_TYPE and "out of memory" in (error_message or "").lower():
        return True
    message = error_message or ""
    return any(marker in message for marker in CUDA_OOM_MARKERS)


@dataclass(frozen=True)
class DeviceMemory:
    uuid: str
    total: float
    free: float
    utilization_percent: int | None = None


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
                    "--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                    "--id=" + ",".join(map(str, self.devices)),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=settings.query_timeout,
            )
            values = {}
            for line in result.stdout.splitlines():
                index, uuid, total, free, utilization = (
                    item.strip() for item in line.split(",")
                )
                total, free = float(total), float(free)
                if (
                    not uuid.startswith("GPU-")
                    or not math.isfinite(total)
                    or not math.isfinite(free)
                    or total <= 0
                    or not 0 <= free <= total
                ):
                    raise ValueError("invalid GPU memory observation")
                # Some devices report N/A for utilization while memory is readable.
                percent = int(utilization) if utilization.isdecimal() else None
                if percent is not None and percent > 100:
                    percent = None
                values[int(index)] = DeviceMemory(uuid, total, free, percent)
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
    """Host RAM and swap in KiB, with optional CPU utilization in percent.

    The scheduler uses available RAM for admission. Swap and CPU utilization
    are reported for observation.
    """

    total_kb: float
    available_kb: float
    swap_total_kb: float
    swap_free_kb: float
    utilization_percent: float | None = None

    @property
    def available_ratio(self) -> float:
        if self.total_kb <= 0:
            return 0.0
        return self.available_kb / self.total_kb

    @property
    def headroom_kb(self) -> float:
        """MemAvailable above the reserve; a non-positive value admits nothing."""
        return self.available_kb - settings.host_reserve_kb

    @property
    def tight(self) -> bool:
        """Whether host RAM cannot take another launch without risk."""
        return self.headroom_kb <= 0


def fits_reserve(host: HostMemory | None, needed_kb: float) -> bool:
    """Whether one more allocation of needed_kb fits above the host reserve.

    Admission reserves the sum of expected peaks instead of trusting a snapshot:
    spending memory that attempts already in flight are still going to ask for is
    what fills the host while every reading along the way looked healthy.
    """
    return host is not None and not host.tight and host.headroom_kb >= needed_kb


class MeminfoMonitor:
    """Read /proc/meminfo for admission and /proc/stat for CPU utilization."""

    def __init__(self, path: Path | None = None):
        self.path = MEMINFO_PATH if path is None else Path(path)
        self._cpu_ticks: tuple[int, int] | None = None

    def _cpu_utilization(self) -> float | None:
        """Percent of aggregate CPU time spent outside idle and I/O wait."""
        try:
            with self.path.with_name("stat").open() as stream:
                fields = stream.readline().split()
            if len(fields) < 5 or fields[0] != "cpu":
                raise ValueError("missing aggregate CPU counters")
            # Guest times are already included in user and nice.
            ticks = [int(field) for field in fields[1:9]]
            total = sum(ticks)
            idle = sum(ticks[3:5])
        except (OSError, ValueError):
            self._cpu_ticks = None
            return None
        previous = self._cpu_ticks
        self._cpu_ticks = total, idle
        if previous is None:
            return None
        elapsed, idle_delta = total - previous[0], idle - previous[1]
        if elapsed <= 0 or not 0 <= idle_delta <= elapsed:
            return None
        return 100 * (elapsed - idle_delta) / elapsed

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
        return HostMemory(
            total, available, swap_total, swap_free, self._cpu_utilization()
        )
