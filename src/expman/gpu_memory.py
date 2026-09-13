"""Optional PyTorch CUDA allocator telemetry from an experiment worker."""

from importlib import import_module
from numbers import Integral
from pathlib import Path

from .storage import PickleSerializer, write_record


def sample() -> dict[str, float] | None:
    """Return the current and peak PyTorch CUDA allocator usage in kilobytes.

    PyTorch is intentionally imported only in a GPU worker, so CPU-only users and
    GPU workloads using another runtime do not acquire a dependency on it.
    """
    try:
        torch = import_module("torch")
    except ModuleNotFoundError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        values = {
            "gpu_allocated_kb": torch.cuda.memory_allocated(),
            "gpu_reserved_kb": torch.cuda.memory_reserved(),
            "gpu_peak_allocated_kb": torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved_kb": torch.cuda.max_memory_reserved(),
        }
    except RuntimeError:
        return None
    if any(not isinstance(value, Integral) or value < 0 for value in values.values()):
        return None
    return {key: value / 1024 for key, value in values.items()}


def watch(record: Path, interval: float, stop) -> None:
    """Publish allocator telemetry until the experiment worker exits."""
    while not stop.wait(interval):
        value = sample()
        if value is not None:
            write_record(record, value, PickleSerializer())
    value = sample()
    if value is not None:
        write_record(record, value, PickleSerializer())
