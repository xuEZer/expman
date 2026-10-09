"""Read-only views of a running Batch, shaped for the dashboard.

Every value here is either an observation the scheduler already took or
something it already published.  Building a snapshot never fits a model, never
records a sample and never touches admission state, so an open dashboard cannot
change what the scheduler decides.  The whole build is wrapped: a failure in
presentation degrades one payload instead of reaching the experiment.

The scheduler's lock is held only long enough to copy the state it publishes.
Everything derived from those copies is computed outside it, because a client
that stalls on rendering must never hold up a launch decision.
"""

import math
from time import perf_counter, time

from ..events import ProgressEvent, Status

SCHEMA = 1


def number(value):
    """A finite float, or None for anything that is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number_value = float(value)
    return number_value if math.isfinite(number_value) else None


def age(stamp):
    """Seconds since an observation was taken, or None when unknown."""
    moment = number(stamp)
    return None if moment is None else round(max(0.0, time() - moment), 3)


def _stage_names(batch):
    for experiment in batch.experiments:
        stages = getattr(experiment.pipeline, "stages", None)
        if stages:
            return [getattr(stage, "__name__", "?") for stage in stages]
    return []


def remaining_interval(lower, upper, coverage, samples, remaining_groups):
    """The remaining-time interval every snapshot reports, from raw values."""
    return {
        "lower_seconds": number(lower),
        "upper_seconds": number(upper),
        "coverage": number(coverage),
        "samples": samples,
        "remaining_groups": remaining_groups,
    }


def _remaining(batch):
    """The Batch-level interval the CLI already displays, without fitting one."""
    try:
        estimate = batch._display_estimate()
    except Exception:
        return None
    return remaining_interval(
        estimate.lower_seconds,
        estimate.upper_seconds,
        estimate.coverage,
        estimate.completed_samples,
        estimate.remaining_experiments,
    )


def run_counts(entries, active, queue):
    """Per-run state counts from ``(run_id, last_status)`` pairs.

    A run named in ``active`` counts as running and one in ``queue`` as
    pending, both before its status is read; a run without any attempt yet is
    pending, and a status this dashboard does not recognise folds into pending
    as well, so a record written by another version never breaks the list.
    """
    counts = {"running": 0, "pending": 0, "succeeded": 0, "failed": 0, "cancelled": 0}
    for run_id, last_status in entries:
        if run_id in active:
            state = "running"
        elif run_id in queue or last_status is None:
            state = "pending"
        else:
            state = str(last_status).lower()
        counts[state if state in counts else "pending"] += 1
    return counts


def _last_attempt_status(experiment):
    """The status of a run's latest attempt, or None before its first one."""
    attempts = getattr(experiment, "_attempts", ())
    return attempts[-1].status if attempts else None


def _progress(batch, limit=400):
    """Latest intra-Stage progress per run, from the tail of the recorder's log.

    Only the tail is scanned: a long experiment can record hundreds of
    thousands of events, and the dashboard must not walk them every second.
    A custom recorder without a list of events simply reports no progress.
    """
    events = getattr(batch.recorder, "events", None)
    if not isinstance(events, list):
        return {}
    latest = {}
    for event in events[-limit:]:
        if isinstance(event, ProgressEvent) and isinstance(event.run_id, str):
            latest[event.run_id] = {
                "completed": event.completed,
                "total": event.total,
                "unit": event.unit,
                "stage_id": event.stage_id,
            }
    return latest


def _workers(scheduler, running_info, names, progress, now):
    """One entry per running attempt: what it holds, what it was charged, its span."""
    workers = []
    if scheduler is None:
        return workers
    for run_id, worker in scheduler.workers.items():
        info = running_info.get(run_id) or {}
        elapsed = max(0.0, perf_counter() - worker.started)
        expected = worker.expected
        limit = worker.limit
        stage_index = worker.stage_index
        workers.append(
            {
                "run_id": run_id,
                "run": run_id[:6],
                "stage_index": stage_index,
                "stage": names[stage_index] if stage_index < len(names) else None,
                "attempt": worker.stage_attempt,
                "device": worker.device,
                "uuid": worker.uuid,
                "elapsed_seconds": round(elapsed, 1),
                "started_at": round(now - elapsed, 1),
                "expected": None
                if expected is None
                else {
                    "lower_seconds": number(expected.lower),
                    "center_seconds": number(expected.center),
                    "upper_seconds": number(expected.upper),
                },
                "host": {
                    "reserved_kb": number(worker.host_reserve_kb),
                    "resident_kb": number(worker.resident_kb),
                    "peak_kb": number(worker.peak_kb),
                    "cap_kb": number(getattr(limit, "cap_kb", None)),
                    "capped": bool(getattr(limit, "capped", False)),
                    "reached_cap": bool(worker.cap_hit),
                },
                "gpu": {
                    "budget_kb": number(worker.gpu_reserve_kb),
                    "reserved_kb": number(info.get("gpu_reserved_kb")),
                    "allocated_kb": number(info.get("gpu_allocated_kb")),
                    "peak_reserved_kb": number(info.get("gpu_peak_reserved_kb")),
                    "peak_allocated_kb": number(info.get("gpu_peak_allocated_kb")),
                    "cap_kb": number(worker.gpu_cap_kb),
                },
                "dependencies": None
                if worker.dependencies is None
                else len(worker.dependencies),
                "progress": progress.get(run_id),
            }
        )
    workers.sort(key=lambda item: item["started_at"])
    return workers


def _resources(device_memory, host_memory, workers, plan):
    """Card and host memory: what others hold, what expman holds, what is left."""
    capacity = (plan or {}).get("capacity") or {}
    gpu_left = capacity.get("gpu_left_kb") or {}
    held = {}
    for worker in workers:
        entry = held.setdefault(
            worker["device"], {"actual_kb": 0.0, "allocated_kb": 0.0, "budget_kb": 0.0}
        )
        entry["actual_kb"] += worker["gpu"]["reserved_kb"] or 0.0
        entry["allocated_kb"] += worker["gpu"]["allocated_kb"] or 0.0
        entry["budget_kb"] += worker["gpu"]["budget_kb"] or 0.0
    cards = []
    for device, observation in sorted(device_memory.items()):
        total = number(observation.get("total_kb"))
        free = number(observation.get("free_kb"))
        used = None if total is None or free is None else max(0.0, total - free)
        mine = held.get(device, {})
        cards.append(
            {
                "index": device,
                "uuid": observation.get("uuid"),
                "total_kb": total,
                "free_kb": free,
                "other_kb": None
                if used is None
                else max(0.0, used - (mine.get("actual_kb") or 0.0)),
                "expman_actual_kb": mine.get("actual_kb", 0.0),
                "expman_allocated_kb": mine.get("allocated_kb", 0.0),
                "expman_budget_kb": mine.get("budget_kb", 0.0),
                "launchable_kb": number(gpu_left.get(device)),
                "age_seconds": age(observation.get("observed_at")),
            }
        )
    host_total = number(host_memory.get("total_kb"))
    host_available = number(host_memory.get("available_kb"))
    resident = sum(worker["host"]["resident_kb"] or 0.0 for worker in workers)
    budget = sum(worker["host"]["reserved_kb"] or 0.0 for worker in workers)
    used = (
        None
        if host_total is None or host_available is None
        else max(0.0, host_total - host_available)
    )
    return {
        "devices": cards,
        "host": {
            "total_kb": host_total,
            "available_kb": host_available,
            "used_kb": used,
            "other_kb": None if used is None else max(0.0, used - resident),
            "expman_actual_kb": resident,
            "expman_budget_kb": budget,
            "launchable_kb": number(capacity.get("host_left_kb")),
            "headroom_kb": number(host_memory.get("headroom_kb")),
            "swap_total_kb": number(host_memory.get("swap_total_kb")),
            "swap_free_kb": number(host_memory.get("swap_free_kb")),
            "tight": bool(host_memory.get("tight", False)),
            "admits_next": bool(host_memory.get("admits_next", False)),
            "age_seconds": age(host_memory.get("observed_at")),
        },
    }


def stage_samples(history):
    """How each Stage's samples were obtained: measured, reused or censored."""
    counts = {}
    for entry in history:
        stage_index = entry.get("stage")
        if not isinstance(stage_index, int):
            continue
        bucket = counts.setdefault(
            stage_index, {"measured": 0, "reused": 0, "censored": 0}
        )
        if entry.get("reused"):
            bucket["reused"] += 1
        elif entry.get("status") == Status.SUCCEEDED.value:
            bucket["measured"] += 1
        else:
            bucket["censored"] += 1
    return counts


def stage_record(
    index, name, known, completed, total, unit, samples, approximate=False
):
    """One Stage's row in a snapshot, shared by both reading paths.

    ``known`` is False until the scheduler has reported the Stage at all, and
    ``unit`` is the scheduler's unit interval, which only a live batch holds.
    The archived path passes nothing there and marks the row approximate
    instead, so both paths serialize the same keys.
    """
    unit = unit or {}
    return {
        "index": index,
        "name": name,
        "known": known,
        "groups_total": total,
        "groups_done": completed,
        "groups_remaining": None
        if not known or total is None or completed is None
        else max(0, total - completed),
        "unit_lower_seconds": number(unit.get("unit_lower_seconds")),
        "unit_upper_seconds": number(unit.get("unit_upper_seconds")),
        "unit_samples": unit.get("spans"),
        "samples": samples,
        "approximate": approximate,
    }


def _stages(completion, names, view, samples):
    """Per-Stage remaining group counts plus the unit interval to multiply them by.

    Reported before the scheduler exists as well: the Stage list is known from
    the Pipeline, so the dashboard can show the shape of the batch while its
    first attempt is still being prepared.
    """
    counts = (
        {}
        if completion is None
        else {index: (completed, total) for index, completed, total in completion}
    )
    units = (view or {}).get("stages") or {}
    return [
        stage_record(
            index,
            names[index] if index < len(names) else None,
            index in counts,
            *counts.get(index, (None, None)),
            units.get(index) or units.get(str(index)),
            samples.get(index, {}),
        )
        for index in sorted(set(counts) | set(range(len(names))))
    ]


def gantt_rows(history, names):
    """Every recorded attempt as one gantt row, on the Batch effective clock.

    An entry without ``start_seconds`` predates this field (or came from an
    older batch); it is dropped rather than guessed, so the axis never shows a
    fabricated span.
    """
    rows = []
    for entry in history:
        start = number(entry.get("start_seconds"))
        end = number(entry.get("end_seconds"))
        run_id = entry.get("run_id")
        if start is None or end is None or not isinstance(run_id, str):
            continue
        stage_index = entry.get("stage")
        rows.append(
            {
                "run": run_id[:6],
                "stage_index": stage_index,
                "stage": names[stage_index]
                if isinstance(stage_index, int) and stage_index < len(names)
                else None,
                "attempt": entry.get("attempt"),
                "device": entry.get("device"),
                "start_seconds": round(start, 2),
                "end_seconds": round(end, 2),
                "status": entry.get("status"),
                "reused": bool(entry.get("reused")),
            }
        )
    rows.sort(key=lambda row: (row["start_seconds"], row["run"]))
    return rows


def session_marks(marks):
    """Positions on the clock where a previous invocation ended and another began."""
    return sorted(
        value
        for value in (number(mark) for mark in marks or ())
        if value is not None and value >= 0
    )


def _events(items, limit=40):
    """The tail of the scheduler's own event log, oldest first."""
    return [
        {
            "at": number(item.get("at")),
            "age_seconds": age(item.get("at")),
            "kind": item.get("kind"),
            "run": None
            if not isinstance(item.get("run_id"), str)
            else item["run_id"][:6],
            "stage_index": item.get("stage_index"),
            "attempt": item.get("stage_attempt"),
            "device": item.get("device"),
            "status": item.get("status"),
            "detail": item.get("detail"),
        }
        for item in list(items)[-limit:]
    ]


def build_snapshot(batch) -> dict:
    """One JSON-safe snapshot of a Batch, including a degraded one on failure."""
    try:
        return _gather(batch)
    except Exception as error:
        return {
            "schema": SCHEMA,
            "generated_at": time(),
            "source": "live",
            "degraded": f"{type(error).__name__}: {error}",
        }


def _publish_window(batch, scheduler):
    """Copy everything the dashboard reads, then leave the scheduler's lock.

    Group identity needs the lock (the progress thread already takes it for the
    same call), so it is resolved here too.  Every other derivation in this
    module runs on the copies, outside the lock.
    """
    with batch._state_lock:
        try:
            completion = None if scheduler is None else scheduler.stage_completion()
        except Exception:
            completion = None
        return {
            "queue": list(batch._queue),
            "active": dict(batch._active_gpu),
            "progress": dict(batch._stage_progress),
            "history": list(batch._stage_history),
            "marks": list(batch._session_marks),
            "running_info": dict(batch._gpu_running_info),
            "device_memory": dict(batch._device_memory),
            "host_memory": dict(batch._host_memory),
            "elapsed": batch.elapsed_seconds,
            "view": batch._schedule_view,
            "plan": None if scheduler is None else scheduler.last_plan,
            "events": [] if scheduler is None else list(scheduler.events),
            "completion": completion,
        }


def _gather(batch) -> dict:
    scheduler = batch._scheduler
    now = time()
    window = _publish_window(batch, scheduler)
    names = _stage_names(batch)
    workers = _workers(scheduler, window["running_info"], names, _progress(batch), now)
    resources = _resources(
        window["device_memory"], window["host_memory"], workers, window["plan"]
    )
    stages = _stages(
        window["completion"], names, window["view"], stage_samples(window["history"])
    )
    pending_by_stage = {}
    for run_id in window["queue"]:
        stage_index = window["progress"].get(run_id)
        if isinstance(stage_index, int):
            pending_by_stage[stage_index] = pending_by_stage.get(stage_index, 0) + 1
    running_by_stage = {}
    for worker in workers:
        stage_index = worker["stage_index"]
        running_by_stage[stage_index] = running_by_stage.get(stage_index, 0) + 1
    host = resources["host"]
    return {
        "schema": SCHEMA,
        "generated_at": now,
        "source": "live",
        "batch": {
            "name": batch.output_dir.name,
            "dir": str(batch.output_dir),
            "elapsed_seconds": window["elapsed"],
            "coverage": number(batch.estimate_coverage),
            "max_retries": batch.max_retries,
            "experiments": len(batch.experiments),
            "stages": len(names),
            "counts": run_counts(
                (
                    (experiment.run_id, _last_attempt_status(experiment))
                    for experiment in batch.experiments
                ),
                window["active"],
                window["queue"],
            ),
            "remaining": _remaining(batch),
            "session_marks": session_marks(window["marks"]),
        },
        "stages": stages,
        "workers": workers,
        "gantt": gantt_rows(window["history"], names),
        "resources": resources,
        "schedule": {
            "pending_runs": len(window["queue"]),
            "running_runs": len(window["active"]),
            "pending_by_stage": pending_by_stage,
            "running_by_stage": running_by_stage,
            "gate": "open" if host["admits_next"] and not host["tight"] else "closed",
            "estimate": window["view"],
            "plan": window["plan"],
        },
        "events": _events(window["events"]),
    }
