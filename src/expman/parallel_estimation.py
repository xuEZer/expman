"""Stage-group makespan estimates for the GPU scheduler."""

from .estimation import TimeEstimate


def estimate_parallel(scheduler, coverage):
    """Estimate remaining Stage work rather than whole Experiment attempts.

    Equivalent configurations share one representative Stage. A representative
    already materialized in the prefix cache is absent from the work list; only
    Stage groups that still require a process contribute to the makespan.
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
        return TimeEstimate(None, None, coverage, samples, len(queue))
    groups, active_groups = groups
    if not groups:
        return TimeEstimate(0.0, 0.0, coverage, samples, 0)

    tasks = []
    for group, (run_id, stage_index) in groups.items():
        if group in active_groups:
            details = running[run_id]
            expected = details.get("expected_seconds")
            elapsed = details.get("duration", 0.0)
            if not isinstance(expected, (int, float)) or expected <= 0:
                return TimeEstimate(None, None, coverage, samples, len(groups))
            duration = max(0.0, expected - elapsed)
        else:
            duration = scheduler._stage_duration(run_id, stage_index)
            if duration is None:
                return TimeEstimate(None, None, coverage, samples, len(groups))
        tasks.append((duration, group))

    counts = {
        device: sum(value == device for value in active.values())
        for device in batch.devices
    }
    if host.get("tight", False) or not host.get("admits_next", True):
        counts = {device: count for device, count in counts.items() if count}
    else:
        counts = {device: max(1, count) for device, count in counts.items()}
    if not counts:
        return TimeEstimate(None, None, coverage, samples, len(groups))

    slots = {device: [0.0] * count for device, count in counts.items()}
    occupied = {device: 0 for device in counts}
    for group in active_groups:
        run_id, _stage_index = groups[group]
        device = active[run_id]
        if device not in slots:
            continue
        duration = next(value for value, key in tasks if key == group)
        slots[device][occupied[device]] = duration
        occupied[device] += 1
    for duration, group in sorted(tasks, key=lambda item: item[0], reverse=True):
        if group in active_groups:
            continue
        device = min(slots, key=lambda item: min(slots[item]))
        slot = min(range(len(slots[device])), key=slots[device].__getitem__)
        slots[device][slot] += duration
    makespan = max(max(values) for values in slots.values())
    return TimeEstimate(makespan, makespan, coverage, samples, len(groups))
