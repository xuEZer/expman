"""Batch-owned MPS control daemon and per-worker startup verification."""

import ctypes
import os
import select
import subprocess
import sys
import warnings
from tempfile import TemporaryDirectory

CONTROL = "nvidia-cuda-mps-control"


class MpsService:
    """One private pipe directory for this batch, across all configured GPUs."""

    def __init__(self):
        self.owner = None
        self.pipe = None

    def ready(self, uuids):
        """Start once and poll readiness without stalling the resource tick."""
        if self.owner is None:
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(uuids)}
            # Only the individual worker receives the per-client limit.
            env.pop("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT", None)
            self.owner = subprocess.Popen(
                [sys.executable, __file__],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                env=env,
                start_new_session=True,
            )
        if self.owner.poll() is not None:
            raise RuntimeError(
                "could not start or maintain the batch's NVIDIA MPS control daemon"
            )
        if self.pipe is None:
            assert self.owner.stdout is not None
            if not select.select([self.owner.stdout], [], [], 0)[0]:
                return False
            self.pipe = self.owner.stdout.readline().strip()
            if not self.pipe:
                raise RuntimeError("MPS service owner exited during initialization")
        return True

    def environment(self, device, cap_mib):
        if not self.pipe:
            raise RuntimeError("MPS service is not ready")
        return {
            "CUDA_VISIBLE_DEVICES": device,
            "CUDA_MPS_PIPE_DIRECTORY": self.pipe,
            "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT": f"{device}={cap_mib}M",
        }

    def terminate_client(self, server_pid, client_pid):
        """Quiesce this client's outstanding CUDA work before sending SIGKILL."""
        if self.pipe is None:
            return
        try:
            result = subprocess.run(
                [CONTROL],
                input=f"terminate_client {server_pid} {client_pid}\n",
                env={**os.environ, "CUDA_MPS_PIPE_DIRECTORY": self.pipe},
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            if result.stdout.strip() not in ("", "0"):
                raise RuntimeError(result.stdout.strip())
        except (OSError, subprocess.SubprocessError, RuntimeError) as error:
            warnings.warn(
                f"could not terminate MPS client {client_pid}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )

    def close(self):
        if self.owner is not None:
            assert self.owner.stdin is not None and self.owner.stdout is not None
            self.owner.stdin.close()
            self.owner.wait()
            self.owner.stdout.close()
            self.owner = None
            self.pipe = None


def _own_daemon():
    # The scheduler alone holds the write end. EOF also cleans up on SIGKILL.
    with TemporaryDirectory(prefix="expman-mps-", dir="/tmp") as directory:
        env = {
            **os.environ,
            "CUDA_MPS_PIPE_DIRECTORY": directory,
            "CUDA_MPS_LOG_DIRECTORY": directory,
        }
        subprocess.run(
            [CONTROL, "-d"],
            env=env,
            check=True,
            timeout=15,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )
        try:
            print(directory, flush=True)
            sys.stdin.buffer.read()
        finally:
            subprocess.run(
                [CONTROL],
                input="quit\n",
                text=True,
                env=env,
                check=True,
                timeout=30,
                stdout=subprocess.DEVNULL,
            )


def _retain_primary_context():
    """Keep the primary context alive for the lifetime of this worker."""
    driver = ctypes.CDLL("libcuda.so.1")
    device = ctypes.c_int()
    context = ctypes.c_void_p()
    for name, args in (
        ("cuInit", (0,)),
        ("cuDeviceGet", (ctypes.byref(device), 0)),
        ("cuDevicePrimaryCtxRetain", (ctypes.byref(context), device)),
    ):
        status = getattr(driver, name)(*args)
        if status == 2:
            raise MemoryError(f"CUDA out of memory during MPS startup ({name})")
        if status:
            raise RuntimeError(
                f"MPS client startup failed at {name}: CUDA error {status}"
            )


def verify_client():
    """Check MPS membership before importing or executing user code."""
    _retain_primary_context()
    result = subprocess.run(
        [CONTROL], input="ps\n", capture_output=True, text=True, check=True, timeout=10
    )
    pid = str(os.getpid())
    uuid = os.environ["CUDA_VISIBLE_DEVICES"]
    # R570 prints only a prefix of the UUID in the DEVICE column.
    for line in result.stdout.splitlines():
        fields = line.split()
        if (
            len(fields) >= 4
            and fields[0] == pid
            and fields[2].isdecimal()
            and fields[3].startswith("GPU-")
            and uuid.startswith(fields[3])
        ):
            return int(fields[2])
    raise RuntimeError(f"MPS did not register worker PID {pid} on {uuid}")


if __name__ == "__main__":
    _own_daemon()
