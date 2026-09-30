"""In-process run dashboard for a Batch.

``serve`` starts a loopback HTTP server with a server-sent-events feed of
read-only snapshots, plus a dependency-free page that renders them.  The
dashboard observes the scheduler; it never participates in scheduling.
"""

from .server import DEFAULT_PORT, WebUI, serve
from .snapshot import SCHEMA, build_snapshot

__all__ = ["DEFAULT_PORT", "SCHEMA", "WebUI", "build_snapshot", "serve"]
