"""Process-wide Batch defaults, configured in code with ``expman.set``.

The command line deliberately keeps no flags for retry counts or interval
coverage: import expman next to the Pipeline definition and call ``set()``
there, and every entry point in the process — the CLI included, which imports
the pipeline module — picks the values up.
"""

from .context import _nonnegative_integer
from .estimation import validate_coverage

_SETTINGS: dict[str, float | int] = {"retries": 1, "coverage": 0.8}


def set(**settings: float | int) -> None:
    """Set process-wide Batch defaults.

    ``retries`` is a nonnegative integer (attempts per stage) and
    ``coverage`` is the nominal interval coverage, a number strictly
    between 0 and 1. Unknown keys raise ValueError.
    """
    for key, value in settings.items():
        if key == "retries":
            _nonnegative_integer(value, "retries")  # type: ignore[arg-type]
        elif key == "coverage":
            validate_coverage(value)
        else:
            raise ValueError(f"unknown setting {key!r}; available: {sorted(_SETTINGS)}")
        _SETTINGS[key] = value


def get(key: str) -> float | int:
    """Return the current value of a setting; unknown keys raise ValueError."""
    try:
        return _SETTINGS[key]
    except KeyError:
        raise ValueError(
            f"unknown setting {key!r}; available: {sorted(_SETTINGS)}"
        ) from None
