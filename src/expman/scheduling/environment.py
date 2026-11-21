"""Thread-pool and Hugging Face environment defaults for each worker.

EXPMAN_THREADS overrides the worker_threads setting with a positive integer
or ``off``. If any supported thread-pool variable is already present, the
whole group is inherited from the parent.

An unset EXPMAN_HF_MIRROR or the value ``cn`` selects the hf_mirror setting.
Use ``off`` to skip adding an endpoint, or supply another endpoint directly.
An existing HF_ENDPOINT takes precedence.
"""

import os

from expman.settings import settings

# One environment variable per thread pool. They all have to agree, otherwise a
# worker picks up whichever library sizes its pool last and spawns more threads
# than the budget allows.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
THREADS_OVERRIDE = "EXPMAN_THREADS"
MIRROR_OVERRIDE = "EXPMAN_HF_MIRROR"


def _override(name: str) -> str | None:
    """A nonempty override from the environment, or None when unset."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def thread_budget() -> int | None:
    """Threads one worker may use; None when the ceiling is switched off."""
    override = _override(THREADS_OVERRIDE)
    if override is None:
        return settings.worker_threads
    if override.lower() == "off":
        return None
    try:
        budget = int(override)
    except ValueError:
        raise ValueError(
            f"{THREADS_OVERRIDE} must be 'off' or a positive integer, got {override!r}"
        ) from None
    if budget < 1:
        raise ValueError(
            f"{THREADS_OVERRIDE} must be 'off' or a positive integer, got {override!r}"
        )
    return budget


def hf_endpoint() -> str | None:
    """The Hugging Face endpoint to hand workers; None to leave it alone."""
    override = _override(MIRROR_OVERRIDE)
    if override is None or override.lower() == "cn":
        return settings.hf_mirror
    if override.lower() == "off":
        return None
    return override if "://" in override else f"https://{override}"


def validate() -> None:
    """Check worker environment overrides before creating a new batch directory."""
    thread_budget()
    hf_endpoint()


def overlay() -> dict[str, str]:
    """Values expman adds to a worker environment, shell values winning.

    An exported thread variable switches the whole thread group off: a project
    that sized its own pools has said what it wants, and leaving the remaining
    pools at the framework's number would make the libraries disagree.
    """
    values: dict[str, str] = {}
    budget = thread_budget()
    if budget is not None and not any(name in os.environ for name in THREAD_VARIABLES):
        values.update({name: str(budget) for name in THREAD_VARIABLES})
    endpoint = hf_endpoint()
    if endpoint is not None and "HF_ENDPOINT" not in os.environ:
        values["HF_ENDPOINT"] = endpoint
    return values
