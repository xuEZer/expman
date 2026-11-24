"""Read an active or stopped batch from its published snapshot or durable records."""

import json
import os
from collections import Counter
from contextlib import suppress
from pathlib import Path
from time import time

from expman.observation.live import LIVE_FILE
from expman.observation.snapshot import (
    SCHEMA,
    gantt_rows,
    number,
    remaining_interval,
    session_marks,
    stage_record,
    stage_samples,
)
from expman.observation.status import RecordedBatch
from expman.scheduling.control import is_running
from expman.settings import settings
from expman.storage.base import StorageError


def _latest_snapshot(directory: Path, running: bool, now: float) -> dict | None:
    """Read the live snapshot, preferring newer durable progress when it is stale."""
    try:
        with (directory / LIVE_FILE).open() as stream:
            data = json.load(stream)
            modified = os.fstat(stream.fileno()).st_mtime_ns
        if not isinstance(data, dict) or not isinstance(data.get("batch"), dict):
            return None
        stamp = number(data.get("generated_at"))
        if (
            running
            and stamp is not None
            and abs(now - stamp) <= settings.live_stale_seconds
        ):
            data["source"] = "live"
        else:
            data["source"] = "last"
            # A resumed batch can advance with live publishing disabled.
            with suppress(FileNotFoundError):
                if (directory / "batch.pkl").stat().st_mtime_ns > modified:
                    archive = _archived(directory)
                    if archive["source"] == "archive":
                        return archive
        return data
    except (OSError, ValueError):
        return None


def _stage_name(spec: object) -> str | None:
    if not isinstance(spec, dict):
        return None
    qualified = spec.get("class")
    return qualified.rsplit(".", 1)[-1] if isinstance(qualified, str) else None


def _stages(manifest: dict) -> list[dict]:
    """Per-Stage progress as the durable records describe it.

    History identifies completed runs, including cache hits. The saved queue,
    active runs and their stage positions supply the pending and running counts.
    """
    signature = manifest.get("pipeline") or []
    runs = len(manifest.get("experiments") or [])
    history = manifest.get("stage_history") or []
    samples = stage_samples(history)
    progress = manifest.get("stage_progress", {})
    pending = Counter(progress.get(run_id) for run_id in manifest.get("queue", ()))
    active = Counter(progress.get(run_id) for run_id in manifest.get("active_gpu", {}))
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
            (
                min(len(reached.get(index, ())), runs),
                runs,
                active[index],
                pending[index],
            ),
            None,
            samples.get(index, {}),
            approximate=True,
        )
        for index in range(len(signature))
    ]


def _stage_names(signature: list) -> list:
    return [_stage_name(spec) or "Stage" for spec in signature]


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
    empty = {"name": directory.name, "dir": str(directory)}
    try:
        record = RecordedBatch.read(directory)
    except (StorageError, OSError):
        return {
            "source": "missing",
            "batch": empty,
            "stages": [],
            "workers": [],
            "gantt": [],
            "resources": None,
            "schedule": _idle_schedule({}),
            "events": [],
        }
    counts = record.counts
    manifest = record.manifest
    return {
        "source": "archive",
        "batch": {
            **empty,
            "elapsed_seconds": number(record.elapsed_seconds),
            "coverage": number(manifest.get("estimate_coverage")),
            "max_retries": manifest.get("max_retries"),
            "experiments": len(record.runs),
            "stages": record.stage_count,
            "counts": counts,
            "remaining": remaining_interval(record.remaining),
            "session_marks": session_marks(manifest.get("session_marks")),
        },
        "stages": _stages(manifest),
        "workers": [],
        "gantt": gantt_rows(
            manifest.get("stage_history") or [],
            _stage_names(manifest.get("pipeline") or []),
        ),
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
    snapshot = _latest_snapshot(directory, running, now) or _archived(directory)
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
