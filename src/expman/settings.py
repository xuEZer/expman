"""User-configurable defaults shared by the CLI, scheduler and worker processes."""

import math
import tomllib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path

type Value = str | Path | bool | int | float
type Range = tuple[Callable[[float], bool], str]

_POSITIVE: Range = (lambda n: n > 0, "positive")
_NONNEGATIVE: Range = (lambda n: n >= 0, "nonnegative")
_FRACTION: Range = (lambda n: 0 < n <= 1, "greater than 0 and at most 1")


def _number[T: (int, float)](default: T, bounds: Range = _POSITIVE) -> T:
    return field(default=default, metadata={"bounds": bounds})


@dataclass
class Defaults:
    """Built-in values. Durations are seconds and memory budgets are KiB."""

    # Batch creation and CLI
    batches_dir: Path = Path("./batches/")
    pipeline_module: str = "src.pipeline"
    retries: int = _number(1, _NONNEGATIVE)
    coverage: float = _number(0.8, (lambda n: 0 < n < 1, "strictly between 0 and 1"))
    status_interval: float = _number(2.0)  # Terminal status refresh period.

    # Web and observation
    web_host: str = "127.0.0.1"
    web_port: int = _number(8765, (lambda n: 0 <= n <= 65535, "between 0 and 65535"))
    # Consecutive ports to try, starting at web_port.
    web_port_attempts: int = _number(10)
    web_interval: float = _number(
        1.0, (lambda n: 0 < n < 3600, "between 0 and 3600 seconds, exclusive")
    )
    web_heartbeat: float = _number(15.0)  # Maximum wait between SSE frames.
    live_enabled: bool = True
    live_interval: float = _number(1.0)
    live_stale_seconds: float = _number(3.0)
    timing_save_interval: float = _number(5.0)

    # Worker environment and hardware scheduling
    worker_threads: int = _number(4)
    hf_mirror: str = "https://hf-mirror.com"
    # Exclude this much MemAvailable from the scheduler's admission budget.
    host_reserve_kb: int = _number(1024**2, _NONNEGATIVE)
    # Initial per-stage reservations until usable peak samples exist.
    host_peak_kb_default: int = _number(1024**2)
    gpu_peak_kb_default: int = _number(1024**2)
    # Requested memory cap = max(capacity_floor_kb, estimated peak * capacity_factor).
    capacity_factor: float = _number(1.10, (lambda n: n >= 1, "at least 1"))
    capacity_floor_kb: int = _number(512 * 1024)
    # Threshold for classifying a host-memory peak as having reached its limit.
    capacity_near: float = _number(0.9, _FRACTION)
    poll_interval: float = _number(1.0)
    # Delay between cancellations under memory pressure.
    relief_cooldown: float = _number(10.0, _NONNEGATIVE)
    query_timeout: float = _number(3.0)  # Time allowed for nvidia-smi.
    candidates_per_stage: int = _number(32)
    plan_beam_width: int = _number(128)

    # Resource and duration estimates
    peak_quantile: float = _number(0.9, _FRACTION)
    # Raise the estimate after an attempt reaches its cap.
    peak_bump_margin: float = _number(1.5, (lambda n: n > 1, "greater than 1"))
    local_min_samples: int = _number(4)
    local_neighbors: int = _number(8)
    # For n samples, choose neighbor rank max(2, ceil(n / neighborhood)).
    neighborhood: int = _number(8)
    max_columns: int = _number(48)  # Wider feature sets share hashed columns.


settings = Defaults()
_fields = {item.name: item for item in fields(Defaults)}
# Explicit Python overrides take precedence over project defaults, including
# assignments that deliberately select the built-in default.
_python_overrides: dict[str, Value] = {}


def _value(key: str, value: object) -> Value:
    spec = _fields.get(key)
    if spec is None:
        raise ValueError(f"unknown setting {key!r}; available: {', '.join(_fields)}")
    default = spec.default
    if isinstance(default, Path):
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"{key} must be a nonempty path")
        return Path(value)
    if isinstance(default, str):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty string")
        return value
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be true or false")
        return value
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number")
    if isinstance(default, int):
        if not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
    else:
        try:
            value = float(value)
        except OverflowError as error:
            raise ValueError(f"{key} must be finite") from error
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
    allowed, description = spec.metadata["bounds"]
    if not allowed(value):
        raise ValueError(f"{key} must be {description}")
    return value


def set(**overrides: object) -> None:
    """Override named settings in this process. Invalid input leaves all values intact."""
    values = {key: _value(key, value) for key, value in overrides.items()}
    vars(settings).update(values)
    _python_overrides.update(values)


def get(key: str) -> Value:
    """Read the current setting by name, e.g. get("worker_threads")."""
    if key not in _fields:
        raise ValueError(f"unknown setting {key!r}; available: {', '.join(_fields)}")
    return getattr(settings, key)


def _project_values() -> dict[str, Value]:
    path = Path.cwd() / "pyproject.toml"
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as stream:
            tool = tomllib.load(stream).get("tool", {})
        options = tool.get("expman", {}) if isinstance(tool, dict) else {}
        if not isinstance(options, dict):
            raise ValueError("[tool.expman] must be a table")
        return {key: _value(key, value) for key, value in options.items()}
    except (OSError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error


@contextmanager
def project_settings() -> Iterator[None]:
    """Apply project defaults for a CLI invocation and restore the caller's settings."""
    project = _project_values()
    previous = vars(settings).copy()
    explicit = _python_overrides.copy()
    vars(settings).update(project | explicit)
    try:
        yield
    finally:
        vars(settings).update(previous)
        _python_overrides.clear()
        _python_overrides.update(explicit)
