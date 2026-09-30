"""Publish a running Batch's live state into its own directory.

The dashboard runs in a separate process and cannot read the scheduler's
in-memory observations, so the process that owns the Batch writes them out
on a cadence.  The write is atomic and strictly best-effort: a dashboard
that nobody started must not cost an experiment anything, so every failure
here is swallowed rather than raised, and a build that degrades becomes a
degraded file instead of an exception.

The last published file is deliberately *not* removed on exit.  A stopped
batch keeps the final state it actually reached, which is richer and more
truthful than anything the dashboard could reconstruct from disk, and the
reader distinguishes the two by how fresh the file is and whether the PID
record still resolves to a live process.
"""

import json
import os
from contextlib import suppress
from threading import Event, Thread

from .snapshot import build_snapshot

LIVE_FILE = "live.json"
LIVE_INTERVAL = 1.0
LIVE_ENV = "EXPMAN_LIVE"

_DISABLED = {"off", "0", "no", "false"}


def live_enabled() -> bool:
    """Whether this process publishes live state; ``EXPMAN_LIVE=off`` opts out."""
    return os.environ.get(LIVE_ENV, "").strip().lower() not in _DISABLED


class LivePublisher:
    """Write one Batch's live snapshot to its directory on a cadence."""

    def __init__(self, batch, interval: float = LIVE_INTERVAL) -> None:
        self.batch = batch
        self.interval = interval
        self.path = batch.output_dir / LIVE_FILE
        self.stopped = Event()
        self.thread = Thread(target=self._loop, name="expman-live", daemon=True)

    def start(self) -> "LivePublisher":
        self.publish()
        self.thread.start()
        return self

    def close(self) -> None:
        """Stop publishing, leaving the last snapshot behind as final state."""
        self.stopped.set()
        if self.thread.is_alive():
            self.thread.join(timeout=5.0)
        self.publish()

    def _loop(self) -> None:
        while not self.stopped.wait(self.interval):
            self.publish()

    def publish(self) -> None:
        """Replace the snapshot atomically; never raise into the experiment."""
        try:
            payload = json.dumps(build_snapshot(self.batch), default=str)
        except Exception:
            return
        temporary = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(payload)
            os.replace(temporary, self.path)
        except OSError:
            with suppress(OSError):
                temporary.unlink()
