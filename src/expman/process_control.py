"""Linux batch stop requests through a FIFO owned by the running batch."""

import errno
import os
import select
import signal
import stat
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path

from expman.storage.base import StorageError, run_lock

STOP_FIFO = "stop.fifo"


def _remove_fifo(path: Path) -> None:
    # Called while holding the batch lock. Leave ordinary files and symlinks alone.
    with suppress(FileNotFoundError):
        if stat.S_ISFIFO(path.lstat().st_mode):
            path.unlink()


@contextmanager
def _writer(directory: Path):
    path = directory / STOP_FIFO
    fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        if not stat.S_ISFIFO(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, "expected a batch control FIFO", str(path))
        yield fd
    finally:
        os.close(fd)


def is_running(directory: Path) -> bool:
    """A nonblocking writer can open the FIFO only while a reader holds it."""
    try:
        with _writer(directory):
            return True
    except OSError:
        return False


@contextmanager
def stop_on_sigterm(directory: Path):
    """Receive stop requests on the main thread while holding the batch lock."""
    import fcntl

    requested = False

    def stopped(signum, frame):
        nonlocal requested
        if not requested:
            requested = True
            raise KeyboardInterrupt

    path = directory / STOP_FIFO
    with ExitStack() as cleanup:
        previous = signal.signal(signal.SIGTERM, stopped)
        cleanup.callback(signal.signal, signal.SIGTERM, previous)
        _remove_fifo(path)
        os.mkfifo(path, 0o600)
        cleanup.callback(_remove_fifo, path)

        # Linux O_RDWR holds a writer as well. Closing a status probe therefore
        # cannot generate a hangup, and only a stop byte triggers SIGTERM.
        fd = os.open(path, os.O_RDWR | os.O_NONBLOCK | os.O_NOFOLLOW)
        cleanup.callback(os.close, fd)
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETOWN, os.getpid())
        fcntl.fcntl(fd, fcntl.F_SETSIG, signal.SIGTERM)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_ASYNC)
        cleanup.callback(fcntl.fcntl, fd, fcntl.F_SETFL, flags)
        # A writer may already have sent its byte before notifications were armed.
        if select.select([fd], [], [], 0)[0]:
            stopped(None, None)
        yield


def request_stop(directory: Path) -> tuple[bool, str]:
    """Write to the running batch's control channel."""
    try:
        with _writer(directory) as fd:
            os.write(fd, b"x")
    except OSError as error:
        if error.errno in (errno.ENXIO, errno.EPIPE):
            # A killed batch can leave a FIFO without its reader. A new owner
            # holds the same execution lock before it publishes a replacement.
            with suppress(StorageError, OSError), run_lock(directory):
                _remove_fifo(directory / STOP_FIFO)
        elif error.errno != errno.ENOENT:
            return False, f"cannot stop {directory}: {error}"
        return False, f"{directory} is not running"
    return True, "Stop request sent. The batch will save its progress and exit."
