"""Optional PyTorch CUDA allocator telemetry from an experiment worker."""

from importlib import import_module
from numbers import Integral


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


def limit(kilobytes: float) -> None:
    """Apply a per-process PyTorch allocator ceiling when CUDA is available."""
    try:
        torch = import_module("torch")
        if not torch.cuda.is_available():
            return
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(kilobytes * 1024 / total)
    except (ModuleNotFoundError, RuntimeError, AttributeError, ZeroDivisionError):
        return
