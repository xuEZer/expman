"""Run-state classification shared by the CLI, the dashboard and the archive.

Three readers answer the same question from the same recorded shapes -- a
manifest entry, the set of active GPU workers and the pending queue.  Keeping
one implementation here is what makes ``expman status``, the live snapshot and
the off-process archive agree on what a run's state is and how the states add
up.
"""


def last_status(entry: dict):
    """The status of a run's latest recorded attempt, or None before any."""
    attempts = entry.get("attempts") or []
    return attempts[-1].get("status", "") if attempts else None


def run_state(run_id, last_status, active, queue) -> str:
    """Classify one run: an active run wins, then the queue, then its status.

    A run named in ``active`` is running and one in ``queue`` -- or one with no
    attempt yet -- is pending, both before its status is read.  Otherwise the
    recorded status is lowercased; the caller decides how an unrecognised
    status folds.
    """
    if run_id in active:
        return "running"
    if run_id in queue or last_status is None:
        return "pending"
    return str(last_status).lower()


def run_counts(entries, active, queue) -> dict:
    """Per-run state counts from ``(run_id, last_status)`` pairs.

    A run named in ``active`` counts as running and one in ``queue`` as
    pending, both before its status is read; a run without any attempt yet is
    pending, and a status this reader does not recognise folds into pending as
    well, so a record written by another version never breaks the list.
    """
    counts = {"running": 0, "pending": 0, "succeeded": 0, "failed": 0, "cancelled": 0}
    for run_id, status in entries:
        state = run_state(run_id, status, active, queue)
        counts[state if state in counts else "pending"] += 1
    return counts
