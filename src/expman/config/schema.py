"""Experiment configuration paths, keys and batch-wide GPU selection."""

from pathlib import Path

CONFIGS_DIR = Path("configs")
EXPERIMENT_CONFIGS_DIR = CONFIGS_DIR / "experiments"

COMBINES_KEY = "combines"
GRID_KEY = "grid"
SCAN_KEY = "scan"
DEFAULTS_KEY = "defaults"

CONTROL_KEYS = (COMBINES_KEY, GRID_KEY, SCAN_KEY)


class ConfigError(ValueError):
    """Experiment configuration or its YAML source violates a configuration rule."""


def _safe_segment(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value not in (".", "..")
        and not any(character in value for character in ("/", "\\", "\0"))
    )


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
