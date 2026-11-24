"""Opt-in NVIDIA test doubles for subprocess fault-injection tests."""

import os
import subprocess

import pytest


@pytest.fixture(scope="session")
def nvidia_test_tools(tmp_path_factory):
    directory = tmp_path_factory.mktemp("nvidia")
    source = directory / "driver.c"
    source.write_text(
        "int cuInit(unsigned flags) { return 0; }\n"
        "int cuDeviceGet(int *device, int ordinal) { *device = ordinal; return 0; }\n"
        "int cuDevicePrimaryCtxRetain(void **ctx, int device) { *ctx = (void*)1; return 0; }\n"
    )
    subprocess.run(
        ["cc", "-shared", "-fPIC", str(source), "-o", str(directory / "libcuda.so.1")],
        check=True,
    )
    commands = {
        "nvidia-cuda-mps-control": """case "$1" in -d) exit 0;; esac
read -r cmd
case "$cmd" in
ps)
 echo "$PPID" >> "$EXPMAN_TEST_CLIENTS/$CUDA_VISIBLE_DEVICES"
 echo "$PPID 1 2 $CUDA_VISIBLE_DEVICES";;
quit|terminate_client*) exit 0;;
*) exit 1;;
esac
""",
        "nvidia-smi": """case "$1" in
--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu)
 for d in $(echo "${3#--id=}" | tr , ' '); do echo "$d, GPU-test-$d, 16384, 16000, 0"; done;;
--query-compute-apps=pid,used_gpu_memory)
 clients="$EXPMAN_TEST_CLIENTS/${3#--id=}"
 if [ -f "$clients" ]; then
  while read -r pid; do
   if kill -0 "$pid" 2>/dev/null; then echo "$pid, 0"; fi
  done < "$clients"
 fi;;
*) exit 1;;
esac
""",
    }
    for name, script in commands.items():
        path = directory / name
        path.write_text("#!/bin/sh\n" + script)
        path.chmod(0o755)
    return directory


@pytest.fixture(autouse=True)
def cpu_test_environment(request, monkeypatch):
    # RNG tests can initialize every visible GPU while snapshotting torch state.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    if request.node.get_closest_marker("simulated_gpu") is None:
        return
    monkeypatch.setenv("EXPMAN_TEST_CLIENTS", str(request.getfixturevalue("tmp_path")))
    directory = str(request.getfixturevalue("nvidia_test_tools"))
    monkeypatch.setenv("PATH", directory, prepend=os.pathsep)
    monkeypatch.setenv("LD_LIBRARY_PATH", directory, prepend=os.pathsep)
