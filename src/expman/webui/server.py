"""Serve batch observations from a workspace over HTTP and server-sent events.

Workspace discovery registers new batch names in the shared ID database.
The server reads snapshots or durable batch records, serves bundled static
assets, and forwards stop requests to running batches.
"""

import hashlib
import json
import sqlite3
import threading
from collections import OrderedDict
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import monotonic, time
from typing import cast
from urllib.parse import parse_qs, urlparse

from expman.observation.archive import read_batch, summarize
from expman.observation.registry import ensure_ids
from expman.observation.results import read_results
from expman.observation.snapshot import SCHEMA
from expman.process_control import request_stop
from expman.settings import settings
from expman.webui.result_layout import layout_results

STATIC_ROOT = Path(__file__).resolve().parent / "static"
STATIC_FILES = {
    "app.js": "text/javascript; charset=utf-8",
    "batch.html": "text/html; charset=utf-8",
    "index.html": "text/html; charset=utf-8",
    "results.js": "text/javascript; charset=utf-8",
    "results.css": "text/css; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
}


class Workspace:
    """Every batch under the selected batches directory, read from disk on demand."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (settings.batches_dir if root is None else root).resolve()

    def collect(self) -> tuple[list[dict], dict[int, dict]]:
        """Summaries for the list page and full snapshots keyed by batch ID.

        Each refresh checks the batch control FIFO and the latest published
        snapshot. The archive reader uses durable records if the snapshot is
        missing or unreadable, or if a stale snapshot predates the manifest.
        """
        summaries = []
        details: dict[int, dict] = {}
        for batch_id, directory in sorted(ensure_ids(self.root).items()):
            with suppress(Exception):
                snapshot = read_batch(directory, batch_id)
                details[batch_id] = snapshot
                summaries.append(summarize(snapshot))
        return summaries, details

    def directory(self, batch_id: int) -> Path | None:
        return ensure_ids(self.root).get(batch_id)


def _starting_payload() -> str:
    return json.dumps(
        {"schema": SCHEMA, "generated_at": time(), "batches": [], "starting": True}
    )


class Stream:
    """The newest workspace snapshot, rebuilt on a cadence for all readers.

    Each tick produces one overview frame and one detail frame per watched
    batch. Subscribers share the serialized frames.
    """

    def __init__(self, workspace: Workspace, interval: float) -> None:
        self.workspace = workspace
        self.interval = interval
        self.condition = threading.Condition()
        self.generation = 0
        # Subscriptions are counted, one per reader, not flagged per batch.
        # A page that reloads unsubscribes as it goes, and a flag would let
        # that unsubscribe drop the batch the connection after it had just
        # asked for.
        self.watched: dict[int, int] = {}
        self.frames: dict[object, str] = {"": _starting_payload()}
        self.stopped = threading.Event()
        self.thread = threading.Thread(
            target=self._loop, name="expman-webui", daemon=True
        )

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()

    def watch(self, batch_id: int) -> None:
        """Subscribe one reader to a batch's frames."""
        with self.condition:
            self.watched[batch_id] = self.watched.get(batch_id, 0) + 1

    def unwatch(self, batch_id: int) -> None:
        """Release one reader's subscription, keeping the others'."""
        with self.condition:
            remaining = self.watched.get(batch_id, 0) - 1
            if remaining > 0:
                self.watched[batch_id] = remaining
            else:
                self.watched.pop(batch_id, None)

    def _loop(self) -> None:
        while not self.stopped.is_set():
            self.refresh()
            self.stopped.wait(self.interval)

    def refresh(self) -> None:
        """Refresh the overview and watched details.

        Keep a skipped batch's previous detail only while the ID names that directory.
        A failed refresh publishes the error on the overview frame.
        """
        try:
            summaries, details = self.workspace.collect()
            base = {"schema": SCHEMA, "generated_at": time(), "batches": summaries}
            with self.condition:
                watched = set(self.watched)
                previous = self.frames
            frames: dict[object, str] = {"": json.dumps(base, default=str)}
            for batch_id in watched:
                detail = details.get(batch_id)
                if detail is None and batch_id in previous:
                    last = json.loads(previous[batch_id]).get("batch")
                    if last and last["batch"].get("dir") == str(
                        self.workspace.directory(batch_id)
                    ):
                        detail = last
                frames[batch_id] = json.dumps({**base, "batch": detail}, default=str)
        except Exception as error:
            with self.condition:
                previous = dict(self.frames)
            frames = {
                **previous,
                "": json.dumps(
                    {
                        "schema": SCHEMA,
                        "generated_at": time(),
                        "batches": [],
                        "degraded": f"{type(error).__name__}: {error}",
                    }
                ),
            }
        with self.condition:
            self.frames = frames
            self.generation += 1
            self.condition.notify_all()

    def frame(self, key: object) -> str:
        with self.condition:
            return self.frames.get(key) or self.frames[""]

    def wait(self, generation: int, key: object, timeout: float) -> tuple[int, str]:
        """Wait for a newer frame for the requested batch or the overview.

        On timeout, return the current frame or the overview if the requested
        batch has no frame yet.
        """
        with self.condition:
            deadline = monotonic() + timeout
            while True:
                frame = self.frames.get("") if key is None else self.frames.get(key)
                if frame is not None and self.generation > generation:
                    return self.generation, frame
                remaining = deadline - monotonic()
                if remaining <= 0:
                    return self.generation, frame or self.frames[""]
                self.condition.wait(remaining)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "expman-webui"

    def log_message(self, format, *args):
        """Keep the terminal the dashboard was started from readable."""

    @property
    def dashboard(self) -> "DashboardServer":
        """This handler's server, which owns the workspace and the stream."""
        return cast("DashboardServer", self.server)

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            self._get(parsed.path, parse_qs(parsed.query))
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as error:
            with suppress(Exception):
                self._error(500, f"{type(error).__name__}: {error}")

    def do_POST(self):
        try:
            request = urlparse(self.path)
            parts = _parts(request.path)
            if (
                len(parts) != 3
                or parts[:2] != ["api", "stop"]
                or not parts[2].isdigit()
            ):
                self._error(404, "no such endpoint")
            else:
                self._stop(int(parts[2]), parse_qs(request.query).get("name"))
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as error:
            with suppress(Exception):
                self._error(500, f"{type(error).__name__}: {error}")

    def _directory(self, batch_id, names=None):
        directory = self.dashboard.workspace.directory(batch_id)
        if directory is None:
            self._error(404, f"no batch {batch_id}")
        elif names is not None and names != [directory.name]:
            self._error(409, f"batch ID {batch_id} now names a different batch")
            return None
        return directory

    def _stop(self, batch_id, names):
        directory = self._directory(batch_id, names)
        if directory is None:
            return
        ok, message = request_stop(directory)
        self._body({"status": "stopping" if ok else "refused", "message": message})

    def _get(self, path, query):
        parts = _parts(path)
        if path in ("/", "/index.html"):
            self._static("index.html")
        elif (
            len(parts) >= 2
            and parts[0] == "b"
            and parts[1].isdigit()
            and parts[2:] in ([], ["results"])
        ):
            self._static("batch.html")
        elif len(parts) == 2 and parts[0] == "static":
            self._static(parts[1])
        elif path == "/api/batches":
            self._raw(self.dashboard.stream.frame(""))
        elif len(parts) == 3 and parts[:2] == ["api", "batch"] and parts[2].isdigit():
            self._detail(int(parts[2]))
        elif (
            len(parts) == 4
            and parts[:2] == ["api", "batch"]
            and parts[2].isdigit()
            and parts[3] == "results"
        ):
            directory = self._directory(int(parts[2]))
            if directory is not None:
                payload, tag = self.dashboard.results(directory)
                if self.headers.get("If-None-Match") == tag:
                    self.send_response(304)
                    self.send_header("ETag", tag)
                    self.end_headers()
                else:
                    self._raw(payload, etag=tag)
        elif path == "/api/stream":
            self._stream(_batch_key(query))
        elif path == "/api/health":
            self._body({"status": "ok", "schema": SCHEMA})
        else:
            self._error(404, "no such endpoint")

    def _detail(self, batch_id):
        """One batch's full snapshot, built on this thread for a direct hit."""
        directory = self._directory(batch_id)
        if directory is None:
            return
        try:
            self._body(read_batch(directory, batch_id))
        except Exception as error:
            self._error(500, f"{type(error).__name__}: {error}")

    def _static(self, name):
        """Serve an allow-listed asset; the dashboard ships no other files."""
        kind = STATIC_FILES.get(name)
        if kind is None:
            self._error(404, "no such asset")
            return
        try:
            content = (STATIC_ROOT / name).read_bytes()
        except OSError as error:
            self._error(500, f"cannot read {name}: {error}")
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(content)))
        # The dashboard is edited in place during development; never cache it.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _body(self, payload):
        self._raw(json.dumps(payload, default=str))

    def _raw(self, payload, *, etag=None):
        content = payload.encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        if etag is not None:
            self.send_header("ETag", etag)
        self.end_headers()
        self.wfile.write(content)

    def _error(self, code, message):
        content = json.dumps({"error": message}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _stream(self, key):
        """Send the latest frame to one subscriber after updates and timeouts."""
        stream = self.dashboard.stream
        if key is not None:
            stream.watch(key)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self.wfile.write(b"retry: 2000\n\n")
            self.wfile.flush()
            generation = -1
            while not stream.stopped.is_set():
                generation, payload = stream.wait(
                    generation, key, settings.web_heartbeat
                )
                self.wfile.write(f"data: {payload}\n\n".encode())
                self.wfile.flush()
        finally:
            if key is not None:
                stream.unwatch(key)


def _parts(path: str) -> list[str]:
    return [part for part in path.split("/") if part]


def _batch_key(query: dict) -> int | None:
    """The batch a subscriber is watching, if it named one."""
    values = query.get("batch") or []
    if len(values) != 1 or not values[0].isdigit():
        return None
    return int(values[0])


class _BatchResults:
    """Keep the layout and observe metric commits on one read-only connection."""

    def __init__(self):
        self.db: sqlite3.Connection | None = None
        self.database_id = None
        self.stamp = None
        self.inputs = None
        self.response = ("", "")

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        self.database_id = None

    def read(self, directory: Path) -> tuple[str, str]:
        database = directory / "metrics.sqlite3"
        try:
            stat = database.stat()
            database_id = (stat.st_dev, stat.st_ino)
        except FileNotFoundError:
            database_id = None
        if database_id != self.database_id:
            self.close()
            if database_id is not None:
                self.db = sqlite3.connect(
                    f"{database.as_uri()}?mode=ro", uri=True, check_same_thread=False
                )
            self.database_id = database_id

        # The run directories are created together and stay fixed as stages run.
        try:
            runs = (directory / "experiments").stat()
            run_set = (runs.st_dev, runs.st_ino, runs.st_mtime_ns)
        except FileNotFoundError:
            run_set = None
        # Compare data_version only on this connection. Sampling it before
        # reading the metric rows leaves concurrent commits visible on the next poll.
        version = (
            self.db.execute("PRAGMA data_version").fetchone()[0] if self.db else None
        )
        stamp = (run_set, database_id, version)
        if stamp != self.stamp:
            inputs = read_results(directory, self.db)
            # Repeated writes also update attempt metadata and commit, but only
            # the configuration and metric values participate in dependency analysis.
            if inputs != self.inputs:
                table = layout_results(*inputs)
                payload = json.dumps({"name": directory.name, **table})
                tag = '"' + hashlib.sha256(payload.encode()).hexdigest() + '"'
                self.response = payload, tag
                self.inputs = inputs
            self.stamp = stamp
        return self.response


class DashboardServer(ThreadingHTTPServer):
    """The HTTP server plus the snapshot stream it publishes."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, workspace, *, interval):
        self._results_lock = threading.Lock()
        self._tables: OrderedDict[Path, _BatchResults] = OrderedDict()
        super().__init__(address, Handler)
        self.workspace = workspace
        self.stream = Stream(workspace, interval)
        self.host, self.port = self.server_address[0], self.server_address[1]

    def results(self, directory: Path) -> tuple[str, str]:
        with self._results_lock:
            if directory not in self._tables:
                # Bound both memory and open SQLite readers as users visit batches.
                if len(self._tables) >= 16:
                    self._tables.popitem(last=False)[1].close()
                self._tables[directory] = _BatchResults()
            self._tables.move_to_end(directory)
            return self._tables[directory].read(directory)

    def server_close(self):
        super().server_close()
        with self._results_lock:
            for table in self._tables.values():
                table.close()
            self._tables.clear()


class WebUI:
    """A running dashboard over one batches directory."""

    def __init__(self, server: DashboardServer, display_host: str):
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever, name="expman-webui-http", daemon=True
        )
        self._done = threading.Event()
        self.host = server.host
        self.port = server.port
        self.display_host = display_host

    @property
    def url(self) -> str:
        return f"http://{self.display_host}:{self.port}/"

    def start(self) -> "WebUI":
        self._server.stream.start()
        self._thread.start()
        return self

    def refresh(self) -> None:
        """Rebuild the workspace snapshot now instead of waiting for a tick."""
        self._server.stream.refresh()

    def wait(self) -> None:
        """Wait until close() is called or Ctrl+C interrupts the caller."""
        with suppress(KeyboardInterrupt):
            self._done.wait()

    def close(self):
        self._done.set()
        self._server.stream.close()
        with suppress(Exception):
            self._server.shutdown()
        with suppress(Exception):
            self._server.server_close()


def _check(host, port, interval):
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be an integer between 0 and 65535")
    if not isinstance(host, str) or not host:
        raise ValueError("host must be a nonempty string")
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not 0 < float(interval) < 3600
    ):
        raise ValueError(
            "interval must be greater than zero and less than 3600 seconds"
        )


def serve(
    root: Path | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    interval: float | None = None,
) -> WebUI:
    """Start a dashboard over ``root``, using the batches_dir setting when omitted.

    The dashboard runs in daemon threads, so it also disappears with the
    process that started it. Omitted arguments use the current settings.
    """
    host = settings.web_host if host is None else host
    port = settings.web_port if port is None else port
    interval = settings.web_interval if interval is None else interval
    _check(host, port, interval)
    workspace = Workspace(root)
    last = None
    last_port = min(port + settings.web_port_attempts - 1, 65535) if port else 0
    for candidate in range(port, last_port + 1):
        try:
            server = DashboardServer(
                (host, candidate), workspace, interval=float(interval)
            )
        except OSError as error:
            last = error
            continue
        display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        return WebUI(server, display_host).start()
    raise OSError(f"could not bind dashboard in {port}..{last_port}: {last}")
