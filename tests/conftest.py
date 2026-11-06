"""GPU-free integration fixtures shared by the CLI and fork/exec worker tests."""

import os
import subprocess

import pytest


@pytest.fixture(scope="session", autouse=True)
def simulated_nvidia(tmp_path_factory):
    if os.name != "posix":
        yield
        return
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
case "$cmd" in ps) echo "$PPID 1 2 $CUDA_VISIBLE_DEVICES";; quit|terminate_client*) exit 0;; *) exit 1;; esac
""",
        "nvidia-smi": """case "$1" in
--query-gpu=index,uuid,memory.total,memory.free)
 for d in $(echo "${3#--id=}" | tr , ' '); do echo "$d, GPU-test-$d, 16384, 16000"; done;;
--query-compute-apps=pid,used_gpu_memory) echo "$PPID, 0";;
*) exit 1;;
esac
""",
    }
    for name, script in commands.items():
        path = directory / name
        path.write_text("#!/bin/sh\n" + script)
        path.chmod(0o755)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("PATH", str(directory) + os.pathsep + os.environ["PATH"])
        patch.setenv(
            "LD_LIBRARY_PATH",
            str(directory) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""),
        )
        patch.setenv("CUDA_VISIBLE_DEVICES", "")
        yield
