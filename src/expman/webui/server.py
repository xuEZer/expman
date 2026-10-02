"""A dashboard process that watches a runs/ tree instead of one Batch.

It is deliberately independent of every experiment process: it discovers
batches the same way the CLI does, reads whatever each one has left on disk,
and reports a running batch from the snapshot that batch publishes.  Nothing
here may reach back into an experiment, and the only write it can perform is
the stop request, which reuses the PID record ``expman stop`` already owns.

The page and its scripts ship in this package and use no external resource,
so a cluster without internet access can still open it.
"""

import json
import os
import signal
import threading
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import monotonic, time
from typing import cast
from urllib.parse import parse_qs, urlparse

from ..analysis import analysis_payload, fingerprint
from ..registry import ensure_ids
from .archive import PID_FILE, read_batch, summarize
from .snapshot import SCHEMA

STATIC_ROOT = Path(__file__).resolve().parent / "static"
STATIC_FILES = {
    "app.js": "text/javascript; charset=utf-8",
    "batch.html": "text/html; charset=utf-8",
    "index.html": "text/html; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
}
HEARTBEAT_SECONDS = 15.0
# A taken port is not an experiment failure: try the next few numbers.
PORT_ATTEMPTS = 10
DEFAULT_PORT = 8765


class Workspace:
    """Every batch under one runs/ tree, read from disk on demand."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root

    def collect(self) -> tuple[list[dict], dict[int, dict]]:
        """Summaries for the list page and full snapshots keyed by batch ID.

        The durable records are re-read every tick.  That is a few small
        files per batch and the scan runs on the publisher thread, so a
        current view of a batch that died without cleaning up is worth more
        than the reads it costs.
        """
        summaries = []
        details: dict[int, dict] = {}
        for batch_id, directory in sorted(ensure_ids(self.root).items()):
            with suppress(Exception):
                snapshot = read_batch(directory, batch_id)
                details[batch_id] = snapshot
                summaries.append(summarize(snapshot))
        summaries.sort(key=lambda item: item["id"] or 0)
        return summaries, details

    def directory(self, batch_id: int) -> Path | None:
        return ensure_ids(self.root).get(batch_id)


def stop(batch_id: int, directory: Path) -> tuple[bool, str]:
    """Ask the process that owns a batch to stop, via its PID record."""
    try:
        pid = int((directory / PID_FILE).read_text().strip())
    except (OSError, ValueError):
        return False, f"batch {batch_id} is not running (no live PID record)"
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return False, f"process {pid} is gone"
    except PermissionError:
        return False, f"no permission to signal process {pid}"
    return True, f"stop signal sent to {pid}"


def _starting_payload() -> str:
    return json.dumps(
        {"schema": SCHEMA, "generated_at": time(), "batches": [], "starting": True}
    )


class Stream:
    """The newest workspace snapshot, rebuilt on a cadence for all readers.

    One publisher serves every subscriber: a browser must never make the
    machine query nvidia-smi, and a batch's full snapshot is only serialized
    for the pages actually watching it.
    """

    def __init__(self, workspace: Workspace, interval: float) -> None:
        self.workspace = workspace
        self.interval = interval
        self.condition = threading.Condition()
        self.generation = 0
        self.watched: set[int] = set()
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
        with self.condition:
            self.watched.add(batch_id)

    def unwatch(self, batch_id: int) -> None:
        with self.condition:
            self.watched.discard(batch_id)

    def _loop(self) -> None:
        while not self.stopped.is_set():
            self.refresh()
            self.stopped.wait(self.interval)

    def refresh(self) -> None:
        """Rebuild every frame; a build failure becomes a degraded payload."""
        try:
            summaries, details = self.workspace.collect()
            base = {"schema": SCHEMA, "generated_at": time(), "batches": summaries}
            with self.condition:
                watched = set(self.watched)
            frames: dict[object, str] = {"": json.dumps(base, default=str)}
            for batch_id in watched:
                frames[batch_id] = json.dumps(
                    {**base, "batch": details.get(batch_id)}, default=str
                )
        except Exception as error:
            frames = {
                "": json.dumps(
                    {
                        "schema": SCHEMA,
                        "generated_at": time(),
                        "batches": [],
                        "degraded": f"{type(error).__name__}: {error}",
                    }
                )
            }
        with self.condition:
            self.frames = frames
            self.generation += 1
            self.condition.notify_all()

    def frame(self, key: object) -> str:
        with self.condition:
            return self.frames.get(key) or self.frames[""]

    def wait(self, generation: int, key: object, timeout: float) -> tuple[int, str]:
        """Block until a frame for ``key`` is newer than ``generation``.

        A subscriber that has just named a batch has to wait for the next
        publish, because a single batch's snapshot is only serialized for the
        pages that asked for it.
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
            parts = _parts(urlparse(self.path).path)
            if (
                len(parts) != 3
                or parts[:2] != ["api", "stop"]
                or not parts[2].isdigit()
            ):
                self._error(404, "no such endpoint")
            elif not self.dashboard.allow_stop:
                self._error(403, "this dashboard is read-only")
            else:
                self._stop(int(parts[2]))
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as error:
            with suppress(Exception):
                self._error(500, f"{type(error).__name__}: {error}")

    def _stop(self, batch_id):
        directory = self.dashboard.workspace.directory(batch_id)
        if directory is None:
            self._error(404, f"no batch {batch_id}")
            return
        ok, message = stop(batch_id, directory)
        self._body({"status": "stopping" if ok else "refused", "message": message})

    def _get(self, path, query):
        parts = _parts(path)
        if path in ("/", "/index.html"):
            self._static("index.html")
        elif len(parts) == 2 and parts[0] == "b" and parts[1].isdigit():
            self._static("batch.html")
        elif len(parts) == 2 and parts[0] == "static":
            self._static(parts[1])
        elif path == "/api/batches":
            self._raw(self.dashboard.stream.frame(""))
        elif len(parts) == 3 and parts[:2] == ["api", "batch"] and parts[2].isdigit():
            self._detail(int(parts[2]))
        elif (
            len(parts) == 3 and parts[:2] == ["api", "analysis"] and parts[2].isdigit()
        ):
            self._analysis(int(parts[2]))
        elif path == "/api/stream":
            self._stream(_batch_key(query))
        elif path == "/api/health":
            self._body(
                {
                    "status": "ok",
                    "schema": SCHEMA,
                    "stop": self.dashboard.allow_stop,
                }
            )
        else:
            self._error(404, "no such endpoint")

    def _detail(self, batch_id):
        """One batch's full snapshot, built on this thread for a direct hit."""
        directory = self.dashboard.workspace.directory(batch_id)
        if directory is None:
            self._error(404, f"no batch {batch_id}")
            return
        try:
            self._body(read_batch(directory, batch_id))
        except Exception as error:
            self._error(500, f"{type(error).__name__}: {error}")

    def _analysis(self, batch_id):
        """One batch's result analysis, computed on demand and cached."""
        directory = self.dashboard.workspace.directory(batch_id)
        if directory is None:
            self._error(404, f"no batch {batch_id}")
            return
        try:
            self._body(self.dashboard.analysis(batch_id, directory))
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

    def _raw(self, payload):
        content = payload.encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
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
        """Push every new snapshot to one subscriber until it goes away."""
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
                generation, payload = stream.wait(generation, key, HEARTBEAT_SECONDS)
                frame = (
                    f"data: {payload}\n\n"
                    if payload is not None
                    else ": keep-alive\n\n"
                )
                self.wfile.write(frame.encode())
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


class DashboardServer(ThreadingHTTPServer):
    """The HTTP server plus the snapshot stream it publishes."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, workspace, *, interval, allow_stop):
        super().__init__(address, Handler)
        self.workspace = workspace
        self.stream = Stream(workspace, interval)
        self.allow_stop = allow_stop
        self.analysis_cache: dict[int, tuple[tuple, dict]] = {}
        self.analysis_lock = threading.Lock()
        self.host, self.port = self.server_address[0], self.server_address[1]

    def analysis(self, batch_id: int, directory: Path) -> dict:
        """The analysis payload for one batch, cached until its inputs change."""
        mark = fingerprint(directory)
        with self.analysis_lock:
            cached = self.analysis_cache.get(batch_id)
            if cached is not None and cached[0] == mark:
                return cached[1]
        payload = analysis_payload(directory, batch_id)
        with self.analysis_lock:
            self.analysis_cache[batch_id] = (mark, payload)
        return payload


class WebUI:
    """A running dashboard over one runs/ tree."""

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
        """Block until interrupted; the caller still has to close() the UI."""
        with suppress(KeyboardInterrupt):
            self._done.wait()

    def close(self):
        self._done.set()
        self._server.stream.close()
        with suppress(Exception):
            self._server.shutdown()
        with suppress(Exception):
            self._server.server_close()


def _check(host, port, interval, allow_stop):
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be an integer between 0 and 65535")
    if not isinstance(host, str) or not host:
        raise ValueError("host must be a nonempty string")
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not 0 < float(interval) < 3600
    ):
        raise ValueError("interval must be a positive finite number of seconds")
    if not isinstance(allow_stop, bool):
        raise ValueError("allow_stop must be a boolean")


def serve(
    root: Path | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    interval: float = 1.0,
    allow_stop: bool = True,
) -> WebUI:
    """Start a dashboard over ``root`` (``runs/`` by default) and return it.

    The dashboard runs in daemon threads, so it also disappears with the
    process that started it.  Binding the loopback address is the default;
    exposing it to a network requires passing a host explicitly.
    """
    _check(host, port, interval, allow_stop)
    workspace = Workspace(root)
    last = None
    for candidate in (0,) if port == 0 else range(port, port + PORT_ATTEMPTS):
        try:
            server = DashboardServer(
                (host, candidate),
                workspace,
                interval=float(interval),
                allow_stop=allow_stop,
            )
        except OSError as error:
            last = error
            continue
        display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        return WebUI(server, display_host).start()
    raise OSError(f"no free port in {port}..{port + PORT_ATTEMPTS - 1}: {last}")
