"""Atomic file operations, serializers and exclusive run locks."""

import os
import pickle
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Protocol


class StorageError(RuntimeError):
    """A durable record could not be written, read, or validated."""


class RecoveryWarning(UserWarning):
    """A recoverable problem occurred while reading or recording execution state."""


class Serializer(Protocol):
    def dump(self, value: Any, stream: BinaryIO) -> None: ...
    def load(self, stream: BinaryIO) -> Any: ...


class PickleSerializer:
    """Load only trusted files in a compatible Python/dependency environment."""

    def dump(self, value: Any, stream: BinaryIO) -> None:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, stream: BinaryIO) -> Any:
        return pickle.load(stream)


def write_record(path: Path, value: Any, serializer: Serializer) -> None:
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            serializer.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception as error:
        raise StorageError(f"cannot save {path}: {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_record(path: Path, serializer: Serializer) -> Any:
    try:
        with path.open("rb") as stream:
            return serializer.load(stream)
    except Exception as error:
        raise StorageError(f"cannot load {path}: {error}") from error


@contextmanager
def run_lock(root: Path) -> Iterator[None]:
    """Hold an exclusive run lock until the file closes or the process exits."""
    import fcntl

    with (root / "run.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise StorageError(f"batch is already running: {root}") from error
        yield
