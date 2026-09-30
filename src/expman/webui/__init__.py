"""Run dashboard for the batches under a runs/ tree.

``expman web`` starts a loopback HTTP server that discovers every batch the
CLI knows, publishes server-sent events of read-only snapshots, and serves a
dependency-free page that renders them.  A batch that is not running is
reported from the records it left behind, so the dashboard outlives every
experiment process it watches.  The dashboard observes; it never takes part
in scheduling.
"""

from .archive import read_batch, summarize
from .live import LIVE_FILE, LivePublisher, live_enabled
from .server import DEFAULT_PORT, WebUI, serve
from .snapshot import SCHEMA, build_snapshot

__all__ = [
    "DEFAULT_PORT",
    "LIVE_FILE",
    "LivePublisher",
    "SCHEMA",
    "WebUI",
    "build_snapshot",
    "live_enabled",
    "read_batch",
    "serve",
    "summarize",
]
