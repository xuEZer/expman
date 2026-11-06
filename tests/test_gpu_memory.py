import os
import subprocess
from unittest.mock import Mock

import pytest

from expman.scheduling import _mps, gpu_memory


@pytest.fixture
def query(monkeypatch):
    command = Mock()
    monkeypatch.setattr(subprocess, "run", command)
    return command


def test_measure_only_this_client_not_the_server_or_other_clients(query):
    query.return_value.stdout = f"1,22\n{os.getpid()},7\n2,151"
    assert gpu_memory.sample("GPU-demo") == 7168
    args = query.call_args.args[0]
    assert (
        "--query-compute-apps=pid,used_gpu_memory" in args and "--id=GPU-demo" in args
    )


@pytest.mark.parametrize("rows", ["", "1,2", "{pid},[N/A]", "{pid},nan"])
def test_missing_measurement_is_unknown(query, rows):
    query.return_value.stdout = rows.format(pid=os.getpid())
    assert gpu_memory.sample("GPU-demo") is None


def test_timeout_is_not_a_zero_reading(query):
    query.side_effect = subprocess.TimeoutExpired("nvidia-smi", 3)
    assert gpu_memory.sample("GPU-demo") is None


def test_mps_verification_requires_our_pid_on_our_device(query, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abcd1234")
    monkeypatch.setattr(_mps, "_retain_primary_context", lambda: None)
    for row in ("", "1 1 2 GPU-abcd", f"{os.getpid()} 1 2 GPU-other"):
        query.return_value.stdout = row
        with pytest.raises(RuntimeError, match="did not register"):
            _mps.verify_client()
    query.return_value.stdout = f"{os.getpid()} 1 2 GPU-abcd"
    _mps.verify_client()
