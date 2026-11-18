"""Pure admission helpers for host and GPU capacity accounting."""

import expman.resources.devices as devices


def _host_growth(workers):
    """Host memory promised to workers above their current resident usage."""
    return sum(
        max(0.0, max(worker.peak_kb, worker.host_reserve_kb) - worker.resident_kb)
        for worker in workers.values()
    )


def _gpu_left(workers, memory, device):
    """Free VRAM after each worker on this card grows to its reservation."""
    growth = sum(
        max(0.0, worker.gpu_reserve_kb - (worker.gpu_used_kb or 0.0))
        for worker in workers.values()
        if worker.device == device
    )
    return memory.free * 1024 - growth


def _device_unbacked(workers, memory, device):
    """Whether one card can no longer honour the contracts running on it.

    Free memory already reflects current use, including use above a reservation.
    Only promised growth is charged against it. An unreadable counter blocks new
    placements on the card but does not identify a victim to shed.
    """
    if any(w.gpu_used_kb is None for w in workers.values() if w.device == device):
        return False
    return _gpu_left(workers, memory[device], device) < 0


def _admits(workers, host, expected_kb):
    """Whether one more attempt of expected_kb fits above the host reserve.

    Every running worker is charged for the gap between what it holds now and
    its reservation, so a launch cannot spend memory that attempts already in
    flight are still going to ask for. Its cgroup ceiling sits above that
    reservation: the ceiling ends an attempt that overshoots rather than being
    capacity a plan may spend twice, and a worker seen to exceed its own
    reservation is charged for the peak it reached.
    """
    return devices.fits_reserve(host, expected_kb + _host_growth(workers))


def _admits_gpu(workers, memory, device, reserve_kb):
    """Reserve only each worker PID's remaining VRAM reservation."""
    return _gpu_left(workers, memory, device) >= reserve_kb


def _resource_left(workers, memory, host):
    """Capacity available after reserving every running worker's future growth."""
    return host.headroom_kb - _host_growth(workers), {
        device: _gpu_left(workers, observation, device)
        for device, observation in memory.items()
    }
