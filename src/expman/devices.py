"""Device selection and whole-device NVIDIA memory observations."""

import math
import subprocess
from dataclasses import dataclass

from .config import ConfigError

MEMORY_MARGIN = 0.1
LAUNCH_INTERVAL = 5.0
POLL_INTERVAL = 0.2
QUERY_TIMEOUT = 10.0
QUERY_RETRY_INTERVAL = 1.0
QUERY_FAILURE_LIMIT = 3


class MemoryObservationError(RuntimeError):
    """A transient query failure; the scheduler may retry without launching work."""


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


class NvidiaMemory:
    def __init__(self, devices):
        self.devices = devices
        self.identities = None

    def sample(self) -> dict[int, DeviceMemory]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
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
