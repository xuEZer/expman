"""Pure admission helpers for host and GPU capacity accounting."""

import expman.config.devices as devices


def _device_unbacked(workers, memory, device):
    """Whether one card can no longer honour the contracts running on it.

    Admission re-reads device memory on every tick, so an external process
    that claims memory after a launch is visible -- but only to the next
    launch, because nothing re-checked the contracts already running. An
    attempt's claim is what it was promised and has not taken yet, plus
    what it took beyond its promise and is therefore spending from another
    attempt's allowance; both come out of what is still free. A claim above
    the free memory is a promise this scheduler can no longer keep.
    """
    claim = sum(
        abs(worker.gpu_reserve_kb - (worker.gpu_reserved_kb or 0.0))
        for worker in workers.values()
        if worker.device == device
    )
    return claim > memory[device].free * 1024


def _admits(workers, host, expected_kb):
    """Whether one more attempt of expected_kb fits above the host reserve.

    Every running worker is charged for the gap between what it holds now and
    its reservation, so a launch cannot spend memory that attempts already in
    flight are still going to ask for. Its cgroup ceiling sits above that
    reservation: the ceiling ends an attempt that overshoots rather than being
    capacity a plan may spend twice, and a worker seen to exceed its own
    reservation is charged for the peak it reached.
    """
    needed = expected_kb
    for _run_id, worker in workers.items():
        reach = max(worker.peak_kb, worker.host_reserve_kb)
        needed += max(0.0, reach - worker.resident_kb)
    return devices.fits_reserve(host, needed)


def _admits_gpu(workers, memory, device, reserve_kb):
    """Reserve only each worker's future PyTorch allocator growth."""
    needed = reserve_kb
    for worker in workers.values():
        if worker.device != device:
            continue
        needed += max(0.0, worker.gpu_reserve_kb - (worker.gpu_reserved_kb or 0))
    return memory.free * 1024 >= needed


def _resource_left(workers, memory, host):
    """Capacity available after reserving every running worker's future growth."""
    host_left = host.headroom_kb
    for worker in workers.values():
        reach = max(worker.peak_kb, worker.host_reserve_kb)
        host_left -= max(0.0, reach - worker.resident_kb)
    gpu_left = {}
    for device, observation in memory.items():
        gpu_left[device] = observation.free * 1024 - sum(
            max(0.0, worker.gpu_reserve_kb - (worker.gpu_reserved_kb or 0))
            for worker in workers.values()
            if worker.device == device
        )
    return host_left, gpu_left
