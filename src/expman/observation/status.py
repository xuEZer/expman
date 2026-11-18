"""Run states and the durable batch overview used by terminal and web views."""

from dataclasses import dataclass
from pathlib import Path

from expman.core.events import Status
from expman.estimation.estimate import TimeEstimate
from expman.storage.base import StorageError
from expman.storage.batch import BatchStore


def last_status(entry: dict):
    """The last attempt's status, or None before a run starts."""
    attempts = entry.get("attempts") or []
    return attempts[-1].get("status") if attempts else None


def run_state(run_id, last_status, active, queue) -> str:
    if run_id in active:
        return "running"
    if run_id in queue or last_status is None:
        return "pending"
    return str(last_status).lower()


def _counts(states) -> dict:
    counts = dict.fromkeys((status.value for status in Status), 0)
    for state in states:
        counts[state if state in counts else "pending"] += 1
    return counts


def run_counts(entries, active, queue) -> dict:
    return _counts(run_state(run_id, last, active, queue) for run_id, last in entries)


@dataclass(frozen=True)
class RecordedRun:
    run_id: str
    state: str
    stage: int | None
    attempts: list


@dataclass(frozen=True)
class RecordedBatch:
    manifest: dict
    runs: list[RecordedRun]
    elapsed_seconds: float | None
    remaining: TimeEstimate | None

    @classmethod
    def read(cls, directory: Path) -> "RecordedBatch":
        store = BatchStore(directory)
        manifest = store.read_manifest()
        active = manifest.get("active_gpu", {})
        queued = set(manifest["queue"])
        progress = manifest.get("stage_progress", {})
        runs = [
            RecordedRun(
                run_id,
                run_state(run_id, last_status(entry), active, queued),
                progress.get(run_id),
                entry.get("attempts", []),
            )
            for entry in manifest["experiments"]
            if isinstance(entry, dict)
            and isinstance(run_id := entry.get("run_id"), str)
        ]
        try:
            elapsed, remaining = store.read_timing() or (None, None)
        except (StorageError, OSError):
            elapsed, remaining = None, None
        return cls(manifest, runs, elapsed, remaining)

    @property
    def counts(self) -> dict:
        return _counts(run.state for run in self.runs)

    @property
    def stage_count(self) -> int:
        return len(self.manifest.get("pipeline", ()))
