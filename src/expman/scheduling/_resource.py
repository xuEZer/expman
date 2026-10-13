"""Pure admission helpers for host and GPU capacity accounting."""

import expman.config.devices as devices


def _device_unbacked(workers, memory, device):
    """Whether one card can no longer honour the contracts running on it."""
    claim = sum(
        abs(worker.gpu_reserve_kb - (worker.gpu_reserved_kb or 0.0))
        for worker in workers.values()
        if worker.device == device
    )
    return claim > memory[device].free * 1024


def _admits(workers, host, expected_kb):
    """Whether one more attempt of expected_kb fits above the host reserve."""
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
