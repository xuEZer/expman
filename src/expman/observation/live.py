"""Publish a batch snapshot at startup, at each interval and during shutdown.

With publishing enabled, the experiment process replaces live.json whether
or not a dashboard is connected. A write failure leaves the previous file
in place. The latest file remains in the batch directory after shutdown.
"""

import json
import os
from contextlib import suppress
from threading import Event, Thread

from expman.observation.snapshot import build_snapshot
from expman.settings import settings

LIVE_FILE = "live.json"
LIVE_ENV = "EXPMAN_LIVE"

_DISABLED = {"off", "0", "no", "false"}


def live_enabled() -> bool:
    """Read the live publishing setting, honoring an explicit environment override."""
    override = os.environ.get(LIVE_ENV, "").strip().lower()
    return override not in _DISABLED if override else settings.live_enabled


class LivePublisher:
    """Write one Batch's live snapshot to its directory on a cadence."""

    def __init__(self, batch, interval: float | None = None) -> None:
        self.batch = batch
        self.interval = settings.live_interval if interval is None else interval
        self.path = batch.output_dir / LIVE_FILE
        self.stopped = Event()
        self.thread = Thread(target=self._loop, name="expman-live", daemon=True)

    def start(self) -> "LivePublisher":
        self.publish()
        self.thread.start()
        return self

    def close(self) -> None:
        """Stop the publishing thread and attempt one final snapshot."""
        self.stopped.set()
        if self.thread.is_alive():
            self.thread.join()
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
