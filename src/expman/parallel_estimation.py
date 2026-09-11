"""Conditional makespan intervals using observed concurrent experiment durations."""

from collections import defaultdict

from ._time_model import DurationModel, features
from .devices import MEMORY_MARGIN
from .estimation import TimeEstimate, _quantile


def estimate_parallel(batch, coverage):
    with batch._state_lock:
        active = dict(batch._active_gpu)
        queue = list(batch._queue)
        history = list(batch._gpu_history)
        running = dict(batch._gpu_running_info)
        memory = dict(batch._gpu_memory)
        host = dict(batch._host_memory)
    pending_ids = set(queue) | set(active)
    completed, censored, pending, _ = batch._time_estimator._observations(pending_ids)
    if not pending:
        return TimeEstimate(0.0, 0.0, coverage, len(completed), 0)
    if not completed:
        return TimeEstimate(None, None, coverage, 0, len(pending))
    counts = {
        device: sum(value == device for value in active.values())
        for device in batch.devices
    }
    capacity = {
        device: max(1, count)
        for device, count in counts.items()
        if count or memory.get(device, 0) >= MEMORY_MARGIN
    }
    if host.get("available_ratio", 1.0) < MEMORY_MARGIN:
        # Host RAM shortage pauses every launch and sheds running attempts, so the
        # forecast keeps only the slots that are already occupied.
        capacity = {device: count for device, count in counts.items() if count}
    if not capacity:
        return TimeEstimate(None, None, coverage, len(completed), len(pending))
    # This forecast conditions on the present slot counts. It does not assume
    # fair GPU sharing: observed wall times and contention are model inputs.
    assignments = dict(active)
    loads = dict(counts)
    for run_id in queue:
        device = min(capacity, key=lambda item: loads[item] / capacity[item])
        assignments[run_id] = device
        loads[device] += 1
    configs = []
    history_by_run = defaultdict(list)
    for item in history:
        history_by_run[item["run_id"]].append(item)
    for experiment in batch.experiments:
        records = history_by_run[experiment.run_id]
        if experiment.run_id in running:
            records.append(running[experiment.run_id])
        duration = sum(item["duration"] for item in records)
        concurrency = (
            sum(item["concurrency"] * item["duration"] for item in records) / duration
            if duration
            else 1.0
        )
        if experiment.run_id in assignments:
            device = assignments[experiment.run_id]
            concurrency = max(1, counts[device])
        else:
            device = records[-1]["device"] if records else -1
        configs.append(
            {
                "experiment": experiment.cfg,
                "execution": {
                    "device": str(device),
                    "concurrency": concurrency,
                },
            }
        )
    model = DurationModel(features(configs), completed, censored)
    ids = [batch.experiments[index].run_id for index, _ in pending]
    totals = []
    for durations in model.draws(pending, components=True):
        predictions = dict(zip(ids, durations, strict=True))
        slots = {device: [0.0] * count for device, count in capacity.items()}
        occupied = {device: 0 for device in capacity}
        for run_id, device in active.items():
            if run_id in predictions:
                slots[device][occupied[device]] = predictions[run_id]
                occupied[device] += 1
        for run_id in queue:
            if run_id not in predictions:
                continue
            device = assignments[run_id]
            slot = min(range(len(slots[device])), key=slots[device].__getitem__)
            slots[device][slot] += predictions[run_id]
        totals.append(max(max(value) for value in slots.values()))
    totals.sort()
    tail = (1 - coverage) / 2
    return TimeEstimate(
        _quantile(totals, tail),
        _quantile(totals, 1 - tail),
        coverage,
        len(completed),
        len(pending),
    )
