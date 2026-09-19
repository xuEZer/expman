"""Device selection plus whole-device and host memory observations."""

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import ConfigError

# Device memory carries no ratio threshold: CUDA OOM is treated as evidence that
# the Stage needs a larger next contract. Signatures come from PyTorch's own
# error type and message.
CUDA_OOM_TYPE = "OutOfMemoryError"
CUDA_OOM_MARKERS = (
    "CUDA out of memory",
    "CUDA error: out of memory",
    "cuDNN error: CUDNN_STATUS_ALLOC_FAILED",
)
# Host RAM kept free of experiment allocations, in kilobytes. Keep the parameter
# available for callers that need a safety margin, while defaulting to a 1 GiB
# scheduler-only reservation; admission still charges each worker's estimated
# peak and cgroup limits continue to protect the host from one runaway worker.
HOST_RESERVE_KB = 1024 * 1024
# Assumed peak resident memory of an attempt that has never been observed, in
# kilobytes. Admission reserves the sum of expected peaks, so this bounds what one
# unknown attempt may hold; over-estimating costs throughput, under-estimating
# costs the host. It is also the cold start of the feature regression.
HOST_PEAK_KB_DEFAULT = 1024 * 1024
# Cold-start PyTorch allocator peak for a top-level Stage. A Stage with completed
# samples that never report PyTorch CUDA usage will subsequently receive no GPU
# contract and can run without occupying a card.
GPU_PEAK_KB_DEFAULT = 1024 * 1024
# Empirical percentile of nearby Stage samples used for admission. It is high
# because the reserve has to bound the next attempt, not describe the average one.
PEAK_QUANTILE = 0.9
# A run stopped by its own cgroup cap comes back with this much more than the peak
# it was seen to reach; the cap is recomputed from the raised estimate.
PEAK_BUMP_MARGIN = 1.5
# Cgroup memory limit written for one worker: the estimate plus room to grow, and
# never below a floor that a Python interpreter with CUDA can start inside. Only
# memory.max is set: memory.high would throttle an allocator that has nothing
# reclaimable to give back (anonymous memory, swap disabled) and stall it instead
# of ending the attempt.
CAPACITY_FACTOR = 1.10
CAPACITY_FLOOR_KB = 512 * 1024
# A worker whose peak came this close to its own limit was stopped by it.
CAPACITY_NEAR = 0.9
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
PROC_ROOT = Path("/proc")
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


def configured_devices(configs) -> tuple[int, ...]:
    selections = []
    for cfg in configs:
        if "device" not in cfg:
            raise ConfigError("device is required for GPU Batch scheduling")
        value = cfg["device"]
        if (
            not isinstance(value, list)
            or not value
            or any(type(item) is not int or item < 0 for item in value)
            or len(set(value)) != len(value)
        ):
            raise ConfigError(
                "device must be a nonempty list of distinct nonnegative GPU indices"
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
    """Host RAM reported by /proc/meminfo; the reading alone stops launches.

    Swap is reported for diagnostics and never gates a launch: occupancy is a
    latch that outlives the shortage which caused it, so it describes history
    rather than the memory available now.
    """

    total_kb: float
    available_kb: float
    swap_total_kb: float
    swap_free_kb: float

    @property
    def available_ratio(self) -> float:
        if self.total_kb <= 0:
            return 0.0
        return self.available_kb / self.total_kb

    @property
    def headroom_kb(self) -> float:
        """MemAvailable above the reserve; a non-positive value admits nothing."""
        return self.available_kb - HOST_RESERVE_KB

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
