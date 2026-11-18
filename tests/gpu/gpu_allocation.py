"""Reserve the least utilized visible GPU for the CUDA/MPS integration tests."""

import ctypes
import errno
import shutil
import socket
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from expman.resources import devices, limits, mps


@dataclass(frozen=True)
class Card:
    index: int
    uuid: str
    utilization: int
    free_mib: int


def _output(command):
    return subprocess.check_output(
        command, text=True, timeout=devices.QUERY_TIMEOUT
    ).splitlines()


def _rank_cards(rows, visible):
    minimum_mib = limits.cap_kb(devices.GPU_PEAK_KB_DEFAULT) / 1024
    cards = []
    for row in rows:
        index, uuid, free, utilization = (field.strip() for field in row.split(","))
        if uuid not in visible or not free.isdecimal() or not utilization.isdecimal():
            continue
        card = Card(int(index), uuid, int(utilization), int(free))
        if card.free_mib >= minimum_mib and card.utilization <= 100:
            cards.append(card)
    return sorted(
        cards, key=lambda card: (card.utilization, -card.free_mib, card.index)
    )


@contextmanager
def reserve_card():
    import torch

    if not sys.platform.startswith("linux") or not shutil.which(mps.CONTROL):
        raise RuntimeError("CUDA/MPS tests require Linux and nvidia-cuda-mps-control")
    if torch.version.cuda is None:
        raise RuntimeError("CUDA/MPS tests require PyTorch built with NVIDIA CUDA")
    if not devices.fits_reserve(
        devices.MeminfoMonitor().sample(), devices.HOST_PEAK_KB_DEFAULT
    ):
        raise RuntimeError("not enough free host RAM for a test worker")

    # A fresh CUDA driver process applies the caller's mask and device ordering,
    # independently of any CUDA initialization during earlier tests.
    visible = set(_output([sys.executable, str(Path(__file__).resolve())]))
    if not visible:
        raise RuntimeError(
            "CUDA reports no visible devices; check CUDA_VISIBLE_DEVICES"
        )
    rows = _output(
        [
            devices.nvidia_smi(),
            "--query-gpu=index,uuid,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    cards = _rank_cards(rows, visible)
    if not cards:
        raise RuntimeError(
            "no visible GPU has usable utilization data and sufficient free VRAM"
        )

    for card in cards:
        # The kernel releases this process's reservation even after SIGKILL.
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as lease:
            try:
                lease.bind(f"\0expman-pytest-{card.uuid}")
            except OSError as error:
                if error.errno == errno.EADDRINUSE:
                    continue
                raise
            yield card
            return
    raise RuntimeError("all eligible GPUs are reserved by concurrent test runs")


def _driver_uuids():
    driver = ctypes.CDLL("libcuda.so.1")

    def check(result):
        if result != 0:
            raise RuntimeError(f"CUDA enumeration failed with driver code {result}")

    check(driver.cuInit(0))
    count = ctypes.c_int()
    check(driver.cuDeviceGetCount(ctypes.byref(count)))
    for ordinal in range(count.value):
        device = ctypes.c_int()
        check(driver.cuDeviceGet(ctypes.byref(device), ordinal))
        raw = (ctypes.c_ubyte * 16)()
        check(driver.cuDeviceGetUuid_v2(ctypes.byref(raw), device))
        yield f"GPU-{UUID(bytes=bytes(raw))}"


if __name__ == "__main__":
    for uuid in _driver_uuids():
        print(uuid)
