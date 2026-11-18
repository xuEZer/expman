"""Process-wide Batch defaults, configured in code with ``expman.set``.

The CLI imports the Pipeline module before constructing a new Batch, so it
sees defaults set in that module. Explicit Batch arguments take precedence.
Resume restores the settings saved in the batch manifest.
"""

from expman.core.context import _nonnegative_integer
from expman.estimation.estimate import validate_coverage

_SETTINGS: dict[str, float | int] = {"retries": 1, "coverage": 0.8}


def set(**settings: float | int) -> None:
    """Set process-wide Batch defaults.

    ``retries`` is a nonnegative integer (extra attempts per failed stage) and
    ``coverage`` is the nominal interval coverage, a number strictly
    between 0 and 1. Unknown keys raise ValueError.
    """
    for key, value in settings.items():
        if key == "retries":
            _nonnegative_integer(value, "retries")
        elif key == "coverage":
            validate_coverage(value)
        else:
            raise ValueError(f"unknown setting {key!r}; available: {sorted(_SETTINGS)}")
    _SETTINGS.update(settings)


def get(key: str) -> float | int:
    """Return the current value of a setting; unknown keys raise ValueError."""
    try:
        return _SETTINGS[key]
    except KeyError:
        raise ValueError(
            f"unknown setting {key!r}; available: {sorted(_SETTINGS)}"
        ) from None
