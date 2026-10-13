"""Process control helpers for stopping a running Batch."""

import os
import signal
from pathlib import Path

PID_FILE = "expman.pid"


def request_stop(directory: Path) -> tuple[bool, str]:
    """Ask the process that owns a batch to stop, via its PID record.

    Returns ``(success, message)``.  The caller decides how to present the
    message and what exit code to use.
    """
    pid_path = directory / PID_FILE
    try:
        pid = int(pid_path.read_text().strip())
    except (OSError, ValueError):
        return False, f"{directory} is not running (no live PID record)"
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_path.unlink(missing_ok=True)
        return False, f"process {pid} is gone; removed the stale PID record"
    except PermissionError:
        return False, f"no permission to signal process {pid}"
    return True, f"Stop signal sent to {pid}; the batch persists and exits shortly."
