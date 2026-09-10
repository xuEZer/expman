"""Atomic trusted-object storage for stage snapshots and checkpoints."""

import math
import os
import pickle
import tempfile
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any, BinaryIO, Protocol


class StorageError(RuntimeError):
    """A durable record could not be written, read, or validated."""


class RecoveryWarning(UserWarning):
    """Recovery fell back from an unreadable checkpoint."""


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


def stage_seconds(record):
    """Return a validated cumulative duration; legacy progress remains unknown."""
    value = record.get("elapsed_seconds")
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise StorageError("invalid cumulative stage duration")
    return value


class RunStore:
    def __init__(self, root: Path, serializer: Serializer | None = None):
        self.root = root
        self.serializer = PickleSerializer() if serializer is None else serializer
        self.rng = None
        self.shared = None
        self.reads = None
        self.metrics = None
        self.run_id = None
        self.cache_parent = None
        self.cache_position = None
        self._timers = {}
        self._saved_seconds = {}

    def start_timing(self, position, checkpoint=None):
        elapsed = 0.0 if checkpoint is None else stage_seconds(checkpoint)
        self._timers[position] = (elapsed, perf_counter())

    def elapsed(self, position):
        timer = self._timers.get(position)
        if timer is None or timer[0] is None:
            return None
        return timer[0] + perf_counter() - timer[1]

    def saved_seconds(self, position):
        return self._saved_seconds.get(position)

    def stop_timing(self, position):
        self._timers.pop(position, None)

    def random_snapshot(self) -> dict:
        return {} if self.rng is None else {"rng_state": self.rng.capture()}

    def restore_random(self, record: dict) -> None:
        if self.rng is None:
            return
        if "rng_state" not in record:
            warnings.warn(
                "legacy snapshot has no random state; exact random replay is unavailable",
                RecoveryWarning,
                stacklevel=2,
            )
            return
        self.rng.restore(record["rng_state"])

    def stage_dir(self, position: tuple[int, ...]) -> Path:
        return self.root.joinpath("stages", *(str(index) for index in position))

    def bind_pipeline(self, position: tuple[int, ...], signature: list) -> None:
        path = self.stage_dir(position) / "pipeline.pkl"
        if path.exists():
            if read_record(path, self.serializer) != signature:
                raise StorageError(f"pipeline definition changed at {path}")
        else:
            write_record(path, signature, self.serializer)

    def status(
        self,
        position,
        status,
        attempt,
        *,
        reused=False,
        elapsed_seconds=None,
        restore_seconds=None,
    ) -> None:
        write_record(
            self.stage_dir(position) / "status.pkl",
            {
                "stage_id": position[-1],
                "position": position,
                "status": status,
                "attempt": attempt,
                "reused": reused,
                "elapsed_seconds": elapsed_seconds,
                "restore_seconds": restore_seconds,
            },
            self.serializer,
        )

    def completed(self, position) -> dict | None:
        path = self.stage_dir(position) / "completed.pkl"
        if not path.exists():
            return None
        record = read_record(path, self.serializer)
        if isinstance(record, dict) and "cache_ref" in record:
            if self.shared is None or record.get("position") != position:
                raise StorageError(f"cannot resolve shared snapshot: {path}")
            reference = record["cache_ref"]
            snapshot = self.shared.load(reference, position)
            snapshot["_cache_ref"] = reference
            snapshot["_shared_reused"] = record["shared_reused"]
            return snapshot
        if (
            not isinstance(record, dict)
            or record.get("position") != position
            or not isinstance(record.get("state"), dict)
            or "output" not in record
            or not isinstance(record.get("name"), str)
        ):
            raise StorageError(f"invalid stage snapshot: {path}")
        stage_seconds(record)
        return record

    def completed_reference(self, position):
        record = read_record(
            self.stage_dir(position) / "completed.pkl", self.serializer
        )
        return record.get("cache_ref")

    def reference(self, position, reference, *, reused):
        write_record(
            self.stage_dir(position) / "completed.pkl",
            {"position": position, "cache_ref": reference, "shared_reused": reused},
            self.serializer,
        )

    def complete(self, position, output, state, name) -> None:
        record = {
            "position": position,
            "name": name,
            "output": output,
            "state": state,
            **self.random_snapshot(),
            "elapsed_seconds": self.elapsed(position),
        }
        if (
            self.shared is not None
            and self.cache_position == position
            and self.cache_parent is not None
        ):
            record["metrics"] = self.metrics.snapshot(self.run_id, position)
            reference = self.shared.publish(self.cache_parent, record, self.reads)
            self.reference(position, reference, reused=False)
        else:
            write_record(
                self.stage_dir(position) / "completed.pkl", record, self.serializer
            )

        self._saved_seconds[position] = record["elapsed_seconds"]

    def checkpoints(self, position) -> list[Path]:
        directory = self.stage_dir(position) / "checkpoints"
        return sorted(directory.glob("[0-9]*.pkl"), reverse=True)

    def latest(self, position, *, restore_random=False) -> dict | None:
        for path in self.checkpoints(position):
            try:
                record = read_record(path, self.serializer)
                if (
                    not isinstance(record, dict)
                    or record.get("position") != position
                    or not isinstance(record.get("state"), dict)
                    or not isinstance(record.get("pipeline_calls", {}), dict)
                    or (
                        record.get("step") is not None
                        and (type(record["step"]) is not int or record["step"] < 0)
                    )
                ):
                    raise StorageError(f"invalid checkpoint: {path}")
                stage_seconds(record)
                if restore_random:
                    self.restore_random(record)
                return record
            except StorageError as error:
                warnings.warn(str(error), RecoveryWarning, stacklevel=2)
        return None


class Checkpoint:
    """Save current state synchronously; keep two successfully written files."""

    def __init__(
        self, store: RunStore, position, state: dict, step=None, pipeline_calls=None
    ):
        self._store = store
        self._position = position
        self._state = state
        self.step = step
        self._pipeline_calls = {} if pipeline_calls is None else pipeline_calls

    def save(self, *, step: int | None = None) -> Path:
        if step is not None and (
            isinstance(step, bool) or not isinstance(step, int) or step < 0
        ):
            raise ValueError("step must be a nonnegative integer or None")
        if self._store.reads is not None:
            self._store.reads.record(())
        existing = self._store.checkpoints(self._position)
        sequence = int(existing[0].stem) + 1 if existing else 1
        path = (
            self._store.stage_dir(self._position)
            / "checkpoints"
            / f"{sequence:020d}.pkl"
        )
        write_record(
            path,
            {
                "position": self._position,
                "step": step,
                "state": self._state,
                "pipeline_calls": self._pipeline_calls,
                **self._store.random_snapshot(),
                "elapsed_seconds": self._store.elapsed(self._position),
            },
            self._store.serializer,
        )
        self.step = step
        for old in self._store.checkpoints(self._position)[2:]:
            old.unlink()
        return path


class RunLock:
    """OS-released local process lock, including when a process is killed."""

    def __init__(self, root: Path):
        self.path = root / "run.lock"
        self.stream = None

    def __enter__(self):
        self.stream = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                self.stream.write(b"0")
                self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            raise StorageError(
                f"batch is already running: {self.path.parent}"
            ) from error
        return self

    def __exit__(self, *args):
        self.stream.close()
