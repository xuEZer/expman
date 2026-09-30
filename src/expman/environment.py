"""Environment values expman sets for every worker process on its own.

Each attempt runs in its own interpreter, launched by the scheduler, and that
process is where user code — and therefore every numerical library — gets
imported. So that is where the framework quietly configures the two things that
are tedious to get right and expensive to get wrong:

* a fixed per-worker thread ceiling, so that N attempts running side by side
  cannot together oversubscribe the CPU, and
* the Hugging Face mirror endpoint, so model downloads do not depend on a
  direct route to huggingface.co.

Nothing here is a public API: the values are merged into the worker environment
at launch, and a project never has to configure them. What the surrounding
shell already exported always wins, and two variables override the defaults:

``EXPMAN_THREADS``
    ``off`` disables the thread ceiling; an integer sets a different one.
``EXPMAN_HF_MIRROR``
    ``off`` leaves ``HF_ENDPOINT`` alone; ``cn`` selects the default mirror;
    any other value is used as the endpoint itself (a scheme is added when the
    value arrives without one).
"""

import os

# Threads one worker may use. Deliberately a fixed number rather than a share of
# the machine: the ceiling exists to bound what many concurrent attempts do to
# the CPU together, so every attempt of a configuration is measured under the
# same budget instead of one whose value depends on how busy the Batch is.
WORKER_THREADS = 4
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
HF_MIRROR = "https://hf-mirror.com"


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
        return WORKER_THREADS
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
        return HF_MIRROR
    if override.lower() == "off":
        return None
    return override if "://" in override else f"https://{override}"


def validate() -> None:
    """Check the overrides the shell carries, before anything has a side effect.

    A malformed value is a configuration mistake, so it is reported while the
    Batch is still being constructed rather than at the first worker launch.
    """
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
