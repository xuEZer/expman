import os
import subprocess
from unittest.mock import Mock

import pytest

from expman.resources import gpu_memory, mps


@pytest.fixture
def query(monkeypatch):
    command = Mock()
    monkeypatch.setattr(subprocess, "run", command)
    return command


def test_measure_only_this_client_not_the_server_or_other_clients(query):
    pid = os.getpid()
    query.return_value.stdout = f"{pid + 1},22\n{pid},7\n{pid + 2},151"
    assert gpu_memory.sample("GPU-demo") == 7168
    args = query.call_args.args[0]
    assert (
        "--query-compute-apps=pid,used_gpu_memory" in args and "--id=GPU-demo" in args
    )


@pytest.mark.parametrize("rows", ["{peer},2", "{pid},[N/A]", "{pid},nan"])
def test_missing_measurement_is_unknown(query, rows):
    pid = os.getpid()
    query.return_value.stdout = rows.format(pid=pid, peer=pid + 1)
    assert gpu_memory.sample("GPU-demo") is None


def test_timeout_is_not_a_zero_reading(query):
    query.side_effect = subprocess.TimeoutExpired("nvidia-smi", 3)
    assert gpu_memory.sample("GPU-demo") is None


def test_mps_verification_requires_our_pid_on_our_device(query, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abcd1234")
    monkeypatch.setattr(mps, "_retain_primary_context", lambda: None)
    pid = os.getpid()
    for row in ("", f"{pid + 1} 1 2 GPU-abcd", f"{pid} 1 2 GPU-other"):
        query.return_value.stdout = row
        with pytest.raises(RuntimeError, match="did not register"):
            mps.verify_client()
    query.return_value.stdout = f"{pid} 1 2 GPU-abcd"
    mps.verify_client()
