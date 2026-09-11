"""Device selection plus whole-device and host memory observations."""

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import ConfigError

# Free-ratio floor shared by device and host memory. At 1% a launch only stops and
# shedding only starts once a resource is nearly exhausted: about 82 MiB of an
# 8 GiB card, or 320 MB of a 32 GiB host, so the OOM margin is left to the user.
MEMORY_MARGIN = 0.01
LAUNCH_INTERVAL = 5.0
POLL_INTERVAL = 0.2
# Measured on RTX 3070 laptop / WSL2, one GPU, at the scheduler's own cadence:
# median 53 ms, p99 300 ms, worst 620 ms with the GPU in desktop use; median
# 143 ms, worst 915 ms with the host CPU saturated. A false timeout kills a live
# attempt and blocks refills, so keep a few times the worst observation.
QUERY_TIMEOUT = 3.0
MEMINFO_PATH = Path("/proc/meminfo")
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
    """Host RAM reported by /proc/meminfo; swap is observed but never gates."""

    total_kb: float
    available_kb: float
    swap_total_kb: float
    swap_free_kb: float

    @property
    def available_ratio(self) -> float:
        return self.available_kb / self.total_kb


class MeminfoMonitor:
    """Read host RAM availability for scheduling; a failed read pauses launches."""

    def __init__(self, path: Path | None = None):
        self.path = MEMINFO_PATH if path is None else Path(path)

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
        return HostMemory(total, available, swap_total, swap_free)
