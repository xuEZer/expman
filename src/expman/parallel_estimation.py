"""Stage-group makespan estimates for the GPU scheduler."""

from time import time

from .estimation import TimeEstimate

_LOWER = 0
_UPPER = 1


def _publish(batch, *, degraded=None, **fields):
    """Record what this pass learned for the dashboard.

    The dashboard is a reader: everything it shows about the schedule is
    published here, by the thread that already did the work, instead of being
    recomputed on demand.  Fitting a duration model is therefore still
    single-writer even while a browser is open.
    """
    view = {
        "at": time(),
        "degraded": degraded,
        "pending_runs": 0,
        "running_runs": 0,
        "groups": None,
        "running_groups": None,
        "samples": 0,
        "device_slots": {},
        "makespan": {"lower_seconds": None, "upper_seconds": None},
        "stages": {},
    }
    view.update(fields)
    batch._schedule_view = view


def _median(values):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _stage_view(groups, active_groups, tasks):
    """Per-Stage group counts and the median representative interval.

    One entry per remaining Stage group, so the dashboard can multiply a
    remaining count by the same unit interval the makespan used.
    """
    stages = {}
    for group, (_run_id, stage_index) in groups.items():
        entry = stages.setdefault(
            stage_index,
            {
                "groups": 0,
                "running": 0,
                "spans": 0,
                "unit_lower_seconds": None,
                "unit_upper_seconds": None,
            },
        )
        entry["groups"] += 1
        if group in active_groups:
            entry["running"] += 1
    measured = {}
    for span, group in tasks:
        measured.setdefault(groups[group][1], []).append(span)
    for stage_index, spans in measured.items():
        entry = stages[stage_index]
        entry["spans"] = len(spans)
        entry["unit_lower_seconds"] = _median([span[_LOWER] for span in spans])
        entry["unit_upper_seconds"] = _median([span[_UPPER] for span in spans])
    return stages


def estimate_parallel(scheduler, coverage):
    """Estimate remaining Stage work rather than whole Experiment attempts.

    Equivalent configurations share one representative Stage. A representative
    already materialized in the prefix cache is absent from the work list; only
    Stage groups that still require a process contribute to the makespan.

    Every task carries an interval, so the packing runs once per side. The two
    makespans bound the same greedy schedule evaluated with the shortest and the
    longest duration each task admits. They are an envelope, not a joint
    confidence interval: the packing adds tasks as if their durations were
    independent, so the two sides are each valid alone rather than together.
    """
    batch = scheduler.batch
    with batch._state_lock:
        queue = list(batch._queue)
        active = dict(batch._active_gpu)
        running = dict(batch._gpu_running_info)
        host = dict(batch._host_memory)
        groups = scheduler._remaining_stage_groups(queue, running)
        samples = sum(
            entry.get("status") == "succeeded"
            and not entry.get("reused")
            and isinstance(entry.get("stage_duration_seconds"), (int, float))
            and entry["stage_duration_seconds"] > 0
            for entry in batch._stage_history
        )
    if groups is None or set(active) != set(running):
        _publish(
            batch,
            degraded="运行中尝试与调度视图不一致",
            pending_runs=len(queue),
            running_runs=len(active),
            samples=samples,
        )
        return TimeEstimate(None, None, coverage, samples, len(queue))
    groups, active_groups = groups
    if not groups:
        # Nothing left to run: keep the per-Stage unit intervals from the last
        # informative pass, so a finished batch still explains how long its
        # Stages took instead of forgetting them.
        previous = (batch._schedule_view or {}).get("stages") or {}
        _publish(
            batch,
            pending_runs=len(queue),
            running_runs=len(active),
            groups=0,
            running_groups=0,
            samples=samples,
            makespan={"lower_seconds": 0.0, "upper_seconds": 0.0},
            stages=previous,
        )
        return TimeEstimate(0.0, 0.0, coverage, samples, 0)

    unknown = TimeEstimate(None, None, coverage, samples, len(groups))
    tasks = []
    incomplete = False
    for group, (run_id, stage_index) in groups.items():
        if group in active_groups:
            elapsed = running[run_id].get("duration", 0.0)
            if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
                incomplete = True
                continue
            span = [
                running[run_id].get(key)
                for key in ("expected_lower_seconds", "expected_upper_seconds")
            ]
            if not all(isinstance(value, (int, float)) and value > 0 for value in span):
                incomplete = True
                continue
            span = [max(0.0, value - elapsed) for value in span]
        else:
            predicted = scheduler._stage_duration(run_id, stage_index)
            if predicted is None:
                incomplete = True
                continue
            span = [predicted.lower, predicted.upper]
        tasks.append((span, group))

    stages = _stage_view(groups, active_groups, tasks)
    if incomplete:
        _publish(
            batch,
            degraded=(
                "尚无阶段耗时样本，无法给出完工区间"
                if not tasks
                else "部分阶段缺少耗时区间，本区间只覆盖已有样本的阶段"
            ),
            pending_runs=len(queue),
            running_runs=len(active),
            groups=len(groups),
            running_groups=len(active_groups),
            samples=samples,
            stages=stages,
        )
        return unknown

    counts = {
        device: sum(value == device for value in active.values())
        for device in batch.devices
    }
    if host.get("tight", False) or not host.get("admits_next", True):
        counts = {device: count for device, count in counts.items() if count}
    else:
        counts = {device: max(1, count) for device, count in counts.items()}
    if not counts:
        _publish(
            batch,
            degraded="主机内存紧张，没有可用的并行位",
            pending_runs=len(queue),
            running_runs=len(active),
            groups=len(groups),
            running_groups=len(active_groups),
            samples=samples,
            stages=stages,
        )
        return unknown

    def makespan(side):
        slots = {device: [0.0] * count for device, count in counts.items()}
        occupied = {device: 0 for device in counts}
        for group in active_groups:
            run_id, _stage_index = groups[group]
            device = active[run_id]
            if device not in slots:
                continue
            duration = next(span[side] for span, key in tasks if key == group)
            slots[device][occupied[device]] = duration
            occupied[device] += 1
        for span, group in sorted(tasks, key=lambda item: item[0][side], reverse=True):
            if group in active_groups:
                continue
            device = min(slots, key=lambda item: min(slots[item]))
            slot = min(range(len(slots[device])), key=slots[device].__getitem__)
            slots[device][slot] += span[side]
        return max(max(values) for values in slots.values())

    lower, upper = makespan(_LOWER), makespan(_UPPER)
    _publish(
        batch,
        pending_runs=len(queue),
        running_runs=len(active),
        groups=len(groups),
        running_groups=len(active_groups),
        samples=samples,
        device_slots=dict(counts),
        makespan={"lower_seconds": lower, "upper_seconds": upper},
        stages=stages,
    )
    return TimeEstimate(lower, upper, coverage, samples, len(groups))
