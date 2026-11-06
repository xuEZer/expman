"""Driver-reported GPU usage for the current MPS client."""

import math
import os
import subprocess

from expman.config import devices


def sample(uuid: str) -> float | None:
    """Current KiB for our PID on our GPU. Missing driver counters stay unknown."""
    try:
        result = subprocess.run(
            [
                devices.nvidia_smi(),
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
                f"--id={uuid}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=devices.QUERY_TIMEOUT,
        )
        pid = str(os.getpid())
        for line in result.stdout.splitlines():
            process, mib = (value.strip() for value in line.split(","))
            if process == pid:
                kib = float(mib) * 1024
                return kib if math.isfinite(kib) and kib >= 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return None
