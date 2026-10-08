"""Report every batch under a runs/ tree without joining its process.

The dashboard is a separate process, so it can only report what a batch
leaves on disk.  Two sources exist and they are merged rather than chosen
between:

* ``live.json``, the snapshot the owning process published while it ran.
  Fresh timestamps plus a live PID record mean the batch is running;
  otherwise the file is the last state the batch actually reached, which
  is richer than anything reconstructible from disk.
* ``batch.pkl`` and ``timing.pkl``, the durable records.  They are the only
  source for batches written before live publishing existed, for batches
  whose publisher was switched off, and for ones that died before the first
  publish.

Both paths produce one shape, so the page never has to know which one it is
looking at.  Reading never fits a model, never imports a Pipeline class and
never executes user code.
"""

import json
import os
from pathlib import Path
from time import time

from ..storage import PickleSerializer, StorageError, read_record
from .live import LIVE_FILE
from .snapshot import (
    SCHEMA,
    number,
    remaining_interval,
    run_counts,
    stage_record,
    stage_samples,
)

STALE_SECONDS = 3.0
PID_FILE = "expman.pid"


def alive(pid: int) -> bool:
    """Whether a PID still resolves to a process we may signal."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def is_running(directory: Path) -> bool:
    """Whether the batch directory holds a PID record for a live process."""
    try:
        raw = (directory / PID_FILE).read_text().strip()
    except OSError:
        return False
    try:
        return alive(int(raw))
    except ValueError:
        return False


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _records(directory: Path) -> tuple[dict | None, dict | None]:
    """The durable manifest and timing records, or None where unreadable."""
    serializer = PickleSerializer()
    manifest = timing = None
    try:
        manifest = read_record(directory / "batch.pkl", serializer)
    except (StorageError, OSError):
        manifest = None
    try:
        timing = read_record(directory / "timing.pkl", serializer)
    except (StorageError, OSError):
        timing = None
    return (
        manifest if isinstance(manifest, dict) else None,
        timing if isinstance(timing, dict) else None,
    )


def _counts(manifest: dict) -> dict:
    return run_counts(
        (
            (entry.get("run_id"), _last_recorded_status(entry))
            for entry in manifest.get("experiments") or []
        ),
        set(manifest.get("active_gpu") or {}),
        set(manifest.get("queue") or []),
    )


def _last_recorded_status(entry: dict):
    """The status of a run's latest recorded attempt, or None before any."""
    attempts = entry.get("attempts") or []
    return attempts[-1].get("status", "") if attempts else None


def _stage_name(spec: object) -> str | None:
    if not isinstance(spec, dict):
        return None
    qualified = spec.get("class")
    return qualified.rsplit(".", 1)[-1] if isinstance(qualified, str) else None


def _stages(manifest: dict) -> list[dict]:
    """Per-Stage progress as the durable records describe it.

    Live group bookkeeping is not reconstructible off-process, so a group is
    approximated by the runs that reached the Stage.  Unit intervals need the
    scheduler's samples, which are live-only, and stay unknown here.
    """
    signature = manifest.get("pipeline") or []
    runs = len(manifest.get("experiments") or [])
    history = manifest.get("stage_history") or []
    samples = stage_samples(history)
    reached: dict[int, set] = {}
    for entry in history:
        index, run_id = entry.get("stage"), entry.get("run_id")
        if not isinstance(index, int) or not isinstance(run_id, str):
            continue
        if entry.get("reused") or entry.get("status") == "succeeded":
            reached.setdefault(index, set()).add(run_id)
    return [
        stage_record(
            index,
            _stage_name(signature[index]),
            True,
            min(len(reached.get(index, ())), runs),
            runs,
            None,
            samples.get(index, {}),
            approximate=True,
        )
        for index in range(len(signature))
    ]


def _remaining(timing: dict | None) -> dict | None:
    estimate = (timing or {}).get("estimate")
    if not isinstance(estimate, dict):
        return None
    return remaining_interval(
        estimate.get("lower_seconds"),
        estimate.get("upper_seconds"),
        estimate.get("coverage"),
        estimate.get("completed_samples"),
        estimate.get("remaining_experiments"),
    )


def _idle_schedule(counts: dict) -> dict:
    return {
        "pending_runs": counts.get("pending", 0),
        "running_runs": counts.get("running", 0),
        "pending_by_stage": {},
        "running_by_stage": {},
        "gate": None,
        "estimate": None,
        "plan": None,
    }


def _archived(directory: Path) -> dict:
    """A snapshot built only from the durable records."""
    manifest, timing = _records(directory)
    empty = {"name": directory.name, "dir": str(directory)}
    if manifest is None:
        return {
            "source": "missing",
            "batch": empty,
            "stages": [],
            "workers": [],
            "resources": None,
            "schedule": _idle_schedule({}),
            "events": [],
        }
    counts = _counts(manifest)
    experiments = manifest.get("experiments") or []
    estimation = manifest.get("estimation") or {}
    return {
        "source": "archive",
        "batch": {
            **empty,
            "elapsed_seconds": number((timing or {}).get("elapsed_seconds")),
            "coverage": number(estimation.get("coverage")),
            "max_retries": manifest.get("max_retries"),
            "experiments": len(experiments),
            "stages": len(manifest.get("pipeline") or []),
            "counts": counts,
            "remaining": _remaining(timing),
        },
        "stages": _stages(manifest),
        "workers": [],
        "resources": None,
        "schedule": _idle_schedule(counts),
        "events": [],
    }


def _status(counts: dict, running: bool, experiments: int) -> str:
    if running:
        return "running"
    if counts.get("failed"):
        return "failed"
    if counts.get("running") or counts.get("pending") or counts.get("cancelled"):
        return "stopped"
    if experiments and counts.get("succeeded") == experiments:
        return "finished"
    return "stopped"


def read_batch(directory: Path, batch_id: int) -> dict:
    """One dashboard snapshot for a batch directory, live or archived."""
    now = time()
    running = is_running(directory)
    live = _read_json(directory / LIVE_FILE)
    if isinstance(live, dict) and isinstance(live.get("batch"), dict):
        snapshot = dict(live)
        stamp = number(live.get("generated_at"))
        fresh = stamp is not None and now - stamp <= STALE_SECONDS
        snapshot["source"] = "live" if (running and fresh) else "last"
    else:
        snapshot = _archived(directory)
    batch = snapshot.get("batch") or {}
    snapshot["id"] = batch_id
    snapshot["running"] = running
    snapshot["status"] = _status(
        batch.get("counts") or {}, running, batch.get("experiments") or 0
    )
    snapshot.setdefault("schema", SCHEMA)
    snapshot.setdefault("generated_at", now)
    stamp = number(snapshot.get("generated_at")) or now
    snapshot["age_seconds"] = round(max(0.0, now - stamp), 3)
    return snapshot


def summarize(snapshot: dict) -> dict:
    """The small per-batch entry the list page renders."""
    batch = snapshot.get("batch") or {}
    return {
        "id": snapshot.get("id"),
        "name": batch.get("name") or "",
        "dir": batch.get("dir") or "",
        "status": snapshot.get("status"),
        "source": snapshot.get("source"),
        "running": bool(snapshot.get("running")),
        "counts": batch.get("counts") or {},
        "elapsed_seconds": batch.get("elapsed_seconds"),
        "coverage": batch.get("coverage"),
        "experiments": batch.get("experiments"),
        "stages": batch.get("stages"),
        "remaining": batch.get("remaining"),
        "workers": len(snapshot.get("workers") or []),
        "age_seconds": snapshot.get("age_seconds"),
        "generated_at": snapshot.get("generated_at"),
        "degraded": snapshot.get("degraded"),
    }
