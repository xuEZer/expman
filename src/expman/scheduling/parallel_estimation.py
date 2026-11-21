"""Stage-group makespan estimates for the GPU scheduler."""

from statistics import median
from time import time

from expman.estimation.duration_model import RUNNING_FIT_COVERAGE, StageDuration
from expman.estimation.estimate import TimeEstimate, valid_seconds

_LOWER = 0
_UPPER = 1


def _publish(batch, coverage, *, degraded=None, **fields):
    """Record what this pass learned for the dashboard.

    The dashboard reads results at the Batch's configured coverage. A caller
    requesting another coverage gets its own TimeEstimate.
    """
    if coverage != batch.estimate_coverage:
        return
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
    batch.live.schedule_view = view


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
        entry["unit_lower_seconds"] = median(span[_LOWER] for span in spans)
        entry["unit_upper_seconds"] = median(span[_UPPER] for span in spans)
    return stages


def _work_span(predicted, running, coverage, now):
    if running is None:
        return (predicted.lower, predicted.upper) if predicted else None
    if predicted is not None:
        elapsed = running.get("duration")
        return (
            predicted.remaining(elapsed, coverage) if valid_seconds(elapsed) else None
        )
    # Without history, estimate this attempt from the pace of its own work.
    forecast = running.get("progress_forecast")
    if forecast is None:
        return None
    reported_at, seconds_left = forecast
    age = max(0.0, now - reported_at)
    return StageDuration(seconds_left, seconds_left, seconds_left).remaining(
        age, coverage
    )


def estimate_parallel(scheduler, coverage):
    """Estimate remaining Stage work rather than whole Experiment attempts.

    Equivalent configurations share one representative Stage. A representative
    already materialized in the prefix cache is absent from the work list; only
    Stage groups that still require a process contribute to the makespan.

    Later Stages wait for their unfinished prefix even when another device is
    idle. Each scenario schedules all remaining groups greedily using one
    endpoint of their predicted durations. The slot assignment and task order
    can differ between the two scenarios.
    """
    batch = scheduler.batch
    with batch._state_lock:
        queue = list(batch.state.queue)
        active = dict(batch.state.active_gpu)
        # A slot is already claimed during worker startup, before the first poll.
        running = {
            run_id: batch.live.running_info.get(
                run_id,
                {"stage_index": batch.state.stage_progress[run_id], "duration": 0.0},
            )
            for run_id in active
        }
        host = dict(batch.live.host_memory)
        groups = scheduler.stages.remaining_groups(queue, running)
        samples = sum(
            entry.get("status") == "succeeded"
            and not entry.get("reused")
            and valid_seconds(entry.get("duration"))
            and entry["duration"] > 0
            for entry in batch.state.stage_history
        )
    if groups is None:
        _publish(
            batch,
            coverage,
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
        previous = (batch.live.schedule_view or {}).get("stages") or {}
        _publish(
            batch,
            coverage,
            pending_runs=len(queue),
            running_runs=len(active),
            groups=0,
            running_groups=0,
            samples=samples,
            makespan={"lower_seconds": 0.0, "upper_seconds": 0.0},
            stages={
                stage: {**details, "groups": 0, "running": 0}
                for stage, details in previous.items()
            },
        )
        return TimeEstimate(0.0, 0.0, coverage, samples, 0)

    unknown = TimeEstimate(None, None, coverage, samples, len(groups))
    tasks = []
    now = time()
    for group, (run_id, stage_index) in groups.items():
        details = running[run_id] if group in active_groups else None
        fit_coverage = coverage if details is None else RUNNING_FIT_COVERAGE
        predicted = scheduler.stages.duration(
            run_id, stage_index, coverage=fit_coverage
        )
        span = _work_span(predicted, details, coverage, now)
        if span is not None and all(valid_seconds(value) for value in span):
            tasks.append((span, group))

    view = {
        "pending_runs": len(queue),
        "running_runs": len(active),
        "groups": len(groups),
        "running_groups": len(active_groups),
        "samples": samples,
        "stages": _stage_view(groups, active_groups, tasks),
    }
    if len(tasks) != len(groups):
        _publish(
            batch,
            coverage,
            degraded="待执行阶段缺少耗时估计，无法给出整批完工区间",
            **view,
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
            coverage,
            degraded="主机内存紧张，没有可用的并行位",
            **view,
        )
        return unknown

    predecessors = {}
    for group, (run_id, stage_index) in groups.items():
        # A shared prefix has the same predecessors for every equivalent run.
        # Completed groups are absent, so find the nearest unfinished one.
        for previous_index in range(stage_index - 1, -1, -1):
            previous = scheduler.stages.group(run_id, previous_index)
            if previous in groups:
                predecessors[group] = previous
                break

    def makespan(side):
        slots = {device: [0.0] * count for device, count in counts.items()}
        occupied = {device: 0 for device in counts}
        durations = {group: span[side] for span, group in tasks}
        finished = {}
        for group in active_groups:
            run_id, _stage_index = groups[group]
            device = active[run_id]
            if device not in slots:
                continue
            duration = durations[group]
            slots[device][occupied[device]] = duration
            occupied[device] += 1
            finished[group] = duration
        pending = {group for _span, group in tasks if group not in active_groups}
        while pending:
            choices = []
            for group in pending:
                previous = predecessors.get(group)
                if previous is not None and previous not in finished:
                    continue
                ready_at = finished.get(previous, 0.0)
                device, slot = min(
                    (
                        (device, slot)
                        for device, times in slots.items()
                        for slot in range(len(times))
                    ),
                    key=lambda item: (
                        max(slots[item[0]][item[1]], ready_at),
                        slots[item[0]][item[1]],
                        item,
                    ),
                )
                start = max(slots[device][slot], ready_at)
                choices.append(
                    (start, -durations[group], repr(group), group, device, slot)
                )
            # Predecessors always have a smaller Stage index, so the graph is
            # acyclic and there is a ready group until all work is assigned.
            start, _priority, _key, group, device, slot = min(choices)
            finish = start + durations[group]
            slots[device][slot] = finish
            finished[group] = finish
            pending.remove(group)
        return max(max(values) for values in slots.values())

    # Longer Stage durations can change greedy placement enough to finish earlier.
    lower, upper = sorted((makespan(_LOWER), makespan(_UPPER)))
    _publish(
        batch,
        coverage,
        device_slots=dict(counts),
        makespan={"lower_seconds": lower, "upper_seconds": upper},
        **view,
    )
    return TimeEstimate(lower, upper, coverage, samples, len(groups))
