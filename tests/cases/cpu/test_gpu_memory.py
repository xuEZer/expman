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


def test_one_query_returns_each_readable_process_on_the_card(query):
    query.return_value.stdout = "21,22\n22,[N/A]\n23,7\n24,nan\n25,0"
    assert gpu_memory.sample("GPU-demo") == {21: 22528, 23: 7168, 25: 0}
    query.assert_called_once()
    args = query.call_args.args[0]
    assert (
        "--query-compute-apps=pid,used_gpu_memory" in args and "--id=GPU-demo" in args
    )


def test_timeout_provides_no_process_readings(query):
    query.side_effect = subprocess.TimeoutExpired("nvidia-smi", 3)
    assert gpu_memory.sample("GPU-demo") == {}


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
