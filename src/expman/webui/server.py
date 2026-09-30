"""In-process HTTP and SSE server for the run dashboard.

The dashboard is a passive observer.  It reads the scheduler's published state,
never fits a model and never touches admission, so a failure anywhere in this
module has to stay invisible to the experiment: every handler is wrapped, the
publisher degrades to an error payload instead of raising, and a port that
cannot be bound is reported to the caller rather than swallowed.

The server binds the loopback address by default and exposes no write route
unless the caller passes ``allow_stop``.
"""

import json
import os
import signal
import threading
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import time
from typing import cast
from urllib.parse import urlparse

from .snapshot import SCHEMA, build_snapshot

STATIC_ROOT = Path(__file__).resolve().parent / "static"
STATIC_FILES = {
    "app.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
    "index.html": "text/html; charset=utf-8",
}
HEARTBEAT_SECONDS = 15.0
# A taken port is not an experiment failure: try the next few numbers.
PORT_ATTEMPTS = 10
DEFAULT_PORT = 8765
STOP_DELAY_SECONDS = 0.25


class Stream:
    """The newest snapshot, rebuilt on a cadence and broadcast to every reader.

    One publisher serves all subscribers: a browser must never make the
    scheduler query nvidia-smi again, because that query costs tens to hundreds
    of milliseconds and the scheduler needs those readings itself.
    """

    def __init__(self, batch, interval):
        self.batch = batch
        self.interval = interval
        self.condition = threading.Condition()
        self.generation = 0
        self.payload = json.dumps(
            {
                "schema": SCHEMA,
                "generated_at": time(),
                "source": "live",
                "starting": True,
            }
        )
        self.stopped = threading.Event()
        self.thread = threading.Thread(
            target=self._loop, name="expman-webui", daemon=True
        )

    def start(self):
        self.thread.start()

    def close(self):
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()

    def _loop(self):
        while not self.stopped.is_set():
            self.refresh()
            self.stopped.wait(self.interval)

    def refresh(self):
        """Rebuild the payload; a build failure becomes a degraded snapshot."""
        try:
            payload = json.dumps(build_snapshot(self.batch), default=str)
        except Exception as error:
            payload = json.dumps(
                {
                    "schema": SCHEMA,
                    "generated_at": time(),
                    "source": "live",
                    "degraded": f"{type(error).__name__}: {error}",
                }
            )
        with self.condition:
            self.payload = payload
            self.generation += 1
            self.condition.notify_all()

    def wait(self, generation, timeout):
        """Block until a newer generation exists, or the timeout elapses."""
        with self.condition:
            if self.generation == generation:
                self.condition.wait(timeout)
            return self.generation, self.payload


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "expman-webui"

    def log_message(self, format, *args):
        """Keep the experiment's own output clean."""

    @property
    def dashboard(self) -> "DashboardServer":
        """This handler's server, which owns the stream and the stop callback."""
        return cast("DashboardServer", self.server)

    def do_GET(self):
        try:
            self._get(urlparse(self.path).path)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as error:
            with suppress(Exception):
                self._error(500, f"{type(error).__name__}: {error}")

    def do_POST(self):
        try:
            path = urlparse(self.path).path
            if path != "/api/stop":
                self._error(404, "no such endpoint")
            elif not self.dashboard.allow_stop:
                self._error(403, "this dashboard is read-only")
            else:
                self._body({"status": "stopping"})
                self._schedule_stop()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as error:
            with suppress(Exception):
                self._error(500, f"{type(error).__name__}: {error}")

    def _schedule_stop(self):
        """Answer first, then stop: the browser must see the acknowledgement."""
        threading.Timer(STOP_DELAY_SECONDS, self.dashboard.request_stop).start()

    def _get(self, path):
        if path in ("/", "/index.html"):
            self._static("index.html")
        elif path.startswith("/static/"):
            self._static(path[len("/static/") :])
        elif path == "/api/snapshot":
            self._raw(self.dashboard.stream.payload)
        elif path == "/api/stream":
            self._stream()
        elif path == "/api/health":
            self._body(
                {"status": "ok", "schema": SCHEMA, "stop": self.dashboard.allow_stop}
            )
        else:
            self._error(404, "no such endpoint")

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
        self._raw(json.dumps(payload))

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

    def _stream(self):
        """Push every new snapshot to one subscriber until it goes away."""
        stream = self.dashboard.stream
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(b"retry: 2000\n\n")
        self.wfile.flush()
        generation = -1
        while not stream.stopped.is_set():
            generation, payload = stream.wait(generation, HEARTBEAT_SECONDS)
            frame = (
                f"data: {payload}\n\n" if payload is not None else ": keep-alive\n\n"
            )
            self.wfile.write(frame.encode())
            self.wfile.flush()


class DashboardServer(ThreadingHTTPServer):
    """The HTTP server plus the snapshot stream it publishes."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, batch, *, interval, allow_stop, on_stop):
        super().__init__(address, Handler)
        self.stream = Stream(batch, interval)
        self.allow_stop = allow_stop
        self.on_stop = on_stop
        self.host, self.port = self.server_address[0], self.server_address[1]
        self.started_at = time()

    def request_stop(self):
        """Ask the owning process to stop; failures stay out of the experiment."""
        if self.on_stop is not None:
            with suppress(Exception):
                self.on_stop()


class WebUI:
    """A running dashboard attached to one Batch."""

    def __init__(self, server: DashboardServer, display_host: str):
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever, name="expman-webui-http", daemon=True
        )
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
        """Rebuild the snapshot now instead of waiting for the next tick."""
        self._server.stream.refresh()

    def close(self):
        self._server.stream.close()
        with suppress(Exception):
            self._server.shutdown()
        with suppress(Exception):
            self._server.server_close()


def _interrupt(signum, frame):
    raise KeyboardInterrupt()


def _install_graceful_stop():
    """Leave an existing SIGTERM handler alone; only replace the default one."""
    with suppress(ValueError, OSError):
        if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
            signal.signal(signal.SIGTERM, _interrupt)


def _default_stop():
    os.kill(os.getpid(), signal.SIGTERM)


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
    batch,
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    interval: float = 1.0,
    allow_stop: bool = False,
    on_stop=None,
) -> WebUI:
    """Start the dashboard for one Batch and return its handle.

    The dashboard runs in daemon threads, so it also disappears with the
    process that owns the Batch.  Binding the loopback address is the default;
    exposing it to a network requires passing a host explicitly.
    """
    _check(host, port, interval, allow_stop)
    if allow_stop and on_stop is None:
        on_stop = _default_stop
        _install_graceful_stop()
    last = None
    for candidate in (0,) if port == 0 else range(port, port + PORT_ATTEMPTS):
        try:
            server = DashboardServer(
                (host, candidate),
                batch,
                interval=float(interval),
                allow_stop=allow_stop,
                on_stop=on_stop,
            )
        except OSError as error:
            last = error
            continue
        display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        return WebUI(server, display_host).start()
    raise OSError(f"no free port in {port}..{port + PORT_ATTEMPTS - 1}: {last}")
