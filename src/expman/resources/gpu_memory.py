"""Driver-reported process memory for one GPU."""

import math
import subprocess

from expman.resources import devices
from expman.settings import settings


def sample(uuid: str) -> dict[int, float]:
    """Return current KiB by PID, omitting processes without a usable reading."""
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
            timeout=settings.query_timeout,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return {}

    readings = {}
    for line in result.stdout.splitlines():
        try:
            process, mib = line.split(",")
            pid = int(process)
            kib = float(mib) * 1024
        except ValueError:
            continue
        if pid > 0 and math.isfinite(kib) and kib >= 0:
            readings[pid] = kib
    return readings
