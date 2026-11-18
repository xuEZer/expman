"""The durable scheduling state of a batch, and the observations of one session."""

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from math import isfinite

from expman.storage.base import StorageError


@dataclass
class LiveObservations:
    host_memory: dict = field(default_factory=dict)
    running_info: dict = field(default_factory=dict)
    device_memory: dict = field(default_factory=dict)
    schedule_view: dict | None = None


@dataclass
class BatchState:
    queue: deque[str]
    stage_progress: dict[str, int]
    stage_elapsed: dict[str, float]
    active_gpu: dict[str, int] = field(default_factory=dict)
    stage_attempts: dict[str, int] = field(default_factory=dict)
    stage_history: list[dict] = field(default_factory=list)
    session_marks: list[float] = field(default_factory=list)

    @classmethod
    def new(cls, run_ids: Iterable[str]) -> "BatchState":
        queue = deque(run_ids)
        return cls(queue, dict.fromkeys(queue, 0), dict.fromkeys(queue, 0.0))

    def next_attempt(self, run_id: str, stage_index: int) -> int:
        return self.stage_attempts.get(f"{run_id}:{stage_index}", 0) + 1

    def _claim(self, run_id: str, stage_index: int) -> int:
        self.queue.remove(run_id)
        attempt = self.next_attempt(run_id, stage_index)
        self.stage_attempts[f"{run_id}:{stage_index}"] = attempt
        return attempt

    def start(self, run_id: str, device: int, stage_index: int) -> int:
        attempt = self._claim(run_id, stage_index)
        self.active_gpu[run_id] = device
        return attempt

    def reuse(self, run_id: str, stage_index: int) -> int:
        """Consume the queued stage without assigning a GPU."""
        return self._claim(run_id, stage_index)

    def rollback_start(self, run_id: str, stage_index: int) -> None:
        self.release(run_id)
        key = f"{run_id}:{stage_index}"
        self.stage_attempts[key] -= 1
        if not self.stage_attempts[key]:
            del self.stage_attempts[key]
        self.queue.appendleft(run_id)

    def release(self, run_id: str) -> None:
        del self.active_gpu[run_id]

    def advance(self, run_id: str) -> None:
        self.stage_progress[run_id] += 1
        self.queue.append(run_id)

    def retry(self, run_id: str) -> None:
        self.queue.append(run_id)

    def failed_attempts(self, run_id: str, stage_index: int) -> int:
        return sum(
            entry["run_id"] == run_id
            and entry["stage"] == stage_index
            and entry["status"] == "failed"
            for entry in self.stage_history
        )

    def recover_active(self) -> list[str]:
        interrupted = list(self.active_gpu)
        self.queue.extendleft(reversed(interrupted))
        self.active_gpu.clear()
        return interrupted

    def to_record(self) -> dict:
        return {
            "queue": list(self.queue),
            "active_gpu": self.active_gpu,
            "stage_progress": self.stage_progress,
            "stage_elapsed": self.stage_elapsed,
            "stage_attempts": self.stage_attempts,
            "stage_history": self.stage_history,
            "session_marks": self.session_marks,
        }

    @classmethod
    def from_record(
        cls, record: dict, run_ids: set[str], devices: tuple[int, ...], stages: int
    ) -> "BatchState":
        """Check identities and cursors before reconstructing any pending work."""
        try:
            state = cls(
                queue=deque(record["queue"]),
                active_gpu=record["active_gpu"],
                stage_progress=record["stage_progress"],
                stage_elapsed=record["stage_elapsed"],
                stage_attempts=record["stage_attempts"],
                stage_history=record["stage_history"],
                session_marks=record["session_marks"],
            )
            queued = set(state.queue)
            active = set(state.active_gpu)
            if (
                len(queued) != len(state.queue)
                or queued & active
                or (queued | active) - run_ids
                or any(
                    type(device) is not int or device not in devices
                    for device in state.active_gpu.values()
                )
            ):
                raise StorageError("invalid pending queue or active GPU workers")
            if (
                state.stage_progress.keys() != run_ids
                or state.stage_elapsed.keys() != run_ids
                or any(
                    type(value) is not int or not 0 <= value < max(1, stages)
                    for value in state.stage_progress.values()
                )
                or any(
                    type(value) not in (float, int) or not isfinite(value) or value < 0
                    for value in state.stage_elapsed.values()
                )
                or not isinstance(state.stage_attempts, dict)
                or not isinstance(state.stage_history, list)
                or not isinstance(state.session_marks, list)
            ):
                raise StorageError("invalid stage progress in batch manifest")
            for key, value in state.stage_attempts.items():
                run_id, stage = key.split(":")
                if (
                    run_id not in run_ids
                    or not 0 <= int(stage) < stages
                    or type(value) is not int
                    or value < 1
                ):
                    raise StorageError("invalid stage attempt in batch manifest")
            for row in state.stage_history:
                if (
                    row["run_id"] not in run_ids
                    or type(row["stage"]) is not int
                    or not 0 <= row["stage"] < stages
                    or row["status"] not in {"succeeded", "failed", "cancelled"}
                ):
                    raise StorageError("invalid stage history in batch manifest")
            return state
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise StorageError("invalid scheduling state in batch manifest") from error
