"""Positional Stage snapshots and checkpoints in one experiment directory."""

import warnings
from pathlib import Path
from sys import float_info
from time import perf_counter
from typing import TYPE_CHECKING, Any

from expman.core.dependencies import validate
from expman.storage.base import (
    PickleSerializer,
    RecoveryWarning,
    Serializer,
    StorageError,
    read_record,
    write_record,
)

if TYPE_CHECKING:
    from expman.core.randomness import RandomStateManager
    from expman.storage.metrics import MetricStore
    from expman.storage.prefix_cache import PrefixCache


def stage_seconds(record: dict) -> float:
    """Read the mandatory cumulative duration saved with a stage's progress."""
    value = record.get("elapsed_seconds")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= value <= float_info.max
    ):
        raise StorageError("invalid cumulative stage duration")
    return value


def validate_snapshot(record, position, *, shared=False) -> None:
    """Validate the common shape of a completed Stage snapshot."""
    if (
        not isinstance(record, dict)
        or record.get("position") != position
        or "state" not in record
        or "output" not in record
        or not isinstance(record.get("name"), str)
        or not isinstance(record.get("config_dependencies"), list)
        or (
            shared
            and (
                "rng_state" not in record or not isinstance(record.get("metrics"), list)
            )
        )
    ):
        raise StorageError("invalid snapshot record")
    stage_seconds(record)


class RunStore:
    def __init__(self, root: Path, serializer: Serializer | None = None):
        self.root = root
        self.serializer = PickleSerializer() if serializer is None else serializer
        self.rng: RandomStateManager | None = None
        self.shared: PrefixCache | None = None
        self.metrics: MetricStore | None = None
        self.run_id: str | None = None

    def random_snapshot(self) -> dict:
        return {} if self.rng is None else {"rng_state": self.rng.capture()}

    def restore_random(self, record: dict) -> None:
        if self.rng is None:
            return
        if "rng_state" not in record:
            raise StorageError("snapshot record has no random state")
        self.rng.restore(record["rng_state"])

    def can_reuse(self, position) -> bool:
        """A shared prefix can supply only work this run has never started."""
        directory = self.stage_dir(position)
        return not (
            (directory / "completed.pkl").exists()
            or (directory / "status.pkl").exists()
            or self.checkpoint_path(position).exists()
        )

    def reuse(self, position, attempt: int, reference, completed: dict) -> None:
        """Attach a shared Stage snapshot and its metrics to this run."""
        assert self.metrics is not None and self.run_id is not None
        self.metrics.restore(self.run_id, attempt, completed["metrics"])
        self.reference(position, reference, reused=True)

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

    def has_completed(self, position) -> bool:
        """Whether this position has a completed record, without reading it.

        The record holds the Stage output, which can be large; callers that only
        need to know that the prefix exists must not pay for it.
        """
        return (self.stage_dir(position) / "completed.pkl").exists()

    def completed(self, position, *, metadata_only=False) -> dict | None:
        """Read a completed Stage. Shared metadata can be read without the output."""
        path = self.stage_dir(position) / "completed.pkl"
        if not path.exists():
            return None
        record = read_record(path, self.serializer)
        if not isinstance(record, dict):
            raise StorageError(f"invalid stage snapshot: {path}")
        if "cache_ref" in record:
            if self.shared is None or record.get("position") != position:
                raise StorageError(f"cannot resolve shared snapshot: {path}")
            reference = record["cache_ref"]
            if metadata_only:
                metadata = self.shared.metadata(reference, position)
                return {
                    "config_dependencies": metadata["dependencies"],
                    "_shared_reused": record["shared_reused"],
                }
            snapshot = self.shared.load(reference, position)
            snapshot["_cache_ref"] = reference
            snapshot["_shared_reused"] = record["shared_reused"]
            return snapshot
        validate_snapshot(record, position)
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

    def complete(self, checkpoint: "Checkpoint", name: str, output: Any, cache_parent):
        position = checkpoint.position
        record = {
            "position": position,
            "name": name,
            "output": output,
            "state": {} if checkpoint.record is None else checkpoint.state,
            "config_dependencies": checkpoint.dependencies,
            **self.random_snapshot(),
            "elapsed_seconds": checkpoint.elapsed(),
        }
        reference = None
        if cache_parent is None:
            write_record(
                self.stage_dir(position) / "completed.pkl", record, self.serializer
            )
        else:
            assert (
                self.shared is not None
                and self.metrics is not None
                and self.run_id is not None
            )
            record["metrics"] = self.metrics.snapshot(self.run_id, position)
            reference = self.shared.publish(cache_parent, record)
            self.reference(position, reference, reused=False)
        self.checkpoint_path(position).unlink(missing_ok=True)
        return reference, record["elapsed_seconds"]

    def checkpoint_path(self, position) -> Path:
        return self.stage_dir(position) / "checkpoint.pkl"

    def latest(self, position, *, dependencies=None) -> dict | None:
        path = self.checkpoint_path(position)
        if not path.exists():
            return None
        try:
            record = read_record(path, self.serializer)
            if (
                not isinstance(record, dict)
                or record.get("position") != position
                or "state" not in record
                or not isinstance(record.get("pipeline_calls"), dict)
                or type(record.get("step")) is not int
                or record["step"] < 0
            ):
                raise StorageError(f"invalid checkpoint: {path}")
            stage_seconds(record)
            saved = record.get("config_dependencies")
            validate(saved)
            if dependencies is not None and saved != dependencies:
                raise StorageError("checkpoint configuration dependencies changed")
            return record
        except (StorageError, ValueError) as error:
            warnings.warn(str(error), RecoveryWarning, stacklevel=2)
            return None


class Checkpoint:
    """Save Stage state synchronously; keep only the newest written file.

    The framework owns one instance per managed Stage; ``state`` is the
    state the user saved through ``ctx.checkpoint()``, restored on the
    next attempt.
    """

    def __init__(
        self,
        store: RunStore,
        position,
        pipeline_calls=None,
        dependencies=None,
        *,
        record: dict | None = None,
    ):
        self._store = store
        self.position = position
        self._pipeline_calls = {} if pipeline_calls is None else pipeline_calls
        self.dependencies = [] if dependencies is None else dependencies
        self.record = record
        if record is not None:
            self._pipeline_calls.update(record["pipeline_calls"])
        self._previous_seconds = 0.0 if record is None else stage_seconds(record)
        self._started = perf_counter()

    @property
    def step(self) -> int:
        return 0 if self.record is None else self.record["step"]

    @property
    def state(self) -> Any:
        return None if self.record is None else self.record["state"]

    def elapsed(self) -> float:
        return self._previous_seconds + perf_counter() - self._started

    def restore_random(self) -> None:
        if self.record is not None:
            self._store.restore_random(self.record)

    def save(self, *, step: int, state: Any) -> Path:
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        path = self._store.checkpoint_path(self.position)
        record = {
            "position": self.position,
            "step": step,
            "state": state,
            "pipeline_calls": self._pipeline_calls,
            "config_dependencies": self.dependencies,
            **self._store.random_snapshot(),
            "elapsed_seconds": self.elapsed(),
        }
        write_record(path, record, self._store.serializer)
        self.record = {"step": step, "state": state}
        return path
