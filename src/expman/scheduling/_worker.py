"""Isolated experiment interpreter; launched only by the GPU scheduler."""

import os
import runpy
import signal
import sys
import threading
import types
from contextlib import suppress
from pathlib import Path
from time import monotonic

from expman.core.context import _error_message
from expman.core.events import Status
from expman.core.experiment import Experiment, StageResult
from expman.scheduling import _mps, gpu_memory
from expman.scheduling.ipc import worker_channel
from expman.scheduling.limits import (
    attempt_cgroup,
    cap_matches_request,
    events,
    memory_cap_kb,
    own_cgroup,
    usage,
)
from expman.storage.base import PickleSerializer, RunLock, read_record

PROC_STATUS = Path("/proc/self/status")


class IpcRecorder:
    def __init__(self, channel):
        self.channel = channel

    def record(self, event):
        self.channel.send({"type": "event", "event": event})


def _own_usage():
    """This process's resident size and high-water mark, in kilobytes.

    Used when the attempt has no cgroup of its own. The enclosing group also
    holds the scheduler and every attempt it has ever started, so reading it
    would charge this one attempt for all of them.
    """
    values = {}
    try:
        for line in PROC_STATUS.read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("VmRSS", "VmHWM"):
                values[key] = float(value.split()[0])
    except (OSError, ValueError, IndexError):
        return None
    if "VmRSS" not in values or "VmHWM" not in values:
        return None
    return values["VmRSS"], values["VmHWM"]


def _resource_snapshot(directory):
    unit = os.environ.get("EXPMAN_CGROUP_UNIT")
    directory = attempt_cgroup(directory, unit)
    host = None if directory is None else usage(directory)
    source = "cgroup"
    diagnostic = None
    actual_cap = memory_cap_kb(directory)
    try:
        requested_cap = float(os.environ.get("EXPMAN_HOST_LIMIT_KB", "0"))
    except ValueError:
        requested_cap = 0.0
    verified = host is not None and cap_matches_request(actual_cap, requested_cap)
    if host is None:
        diagnostic = (
            "independent cgroup counters unreadable"
            if directory is not None
            else "no independent attempt memory cgroup"
        )
        host = _own_usage()
        source = "proc" if host is not None else "unavailable"
        directory = None
        if host is None:
            diagnostic += "; process RSS/HWM unavailable"
    elif not verified and requested_cap > 0:
        diagnostic = "kernel memory limit is missing, unlimited, or exceeds the request"
    return {
        "timestamp": monotonic(),
        "gpu_used_kb": gpu_memory.sample(os.environ["CUDA_VISIBLE_DEVICES"])
        if "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT" in os.environ
        else None,
        "host_current_kb": None if host is None else host[0],
        "host_peak_kb": None if host is None else host[1],
        "host_usage_source": source,
        "host_usage_error": diagnostic,
        "host_limit_kb": actual_cap,
        "host_limit_verified": verified,
        "memory_events": events(directory),
        "cgroup": None if directory is None else str(directory),
    }


def _resource_cgroup():
    """Discover only this exact attempt's memory group, never a shared parent."""
    value = os.environ.get("EXPMAN_CGROUP")
    if value == "":
        return None
    unit = os.environ.get("EXPMAN_CGROUP_UNIT")
    actual = attempt_cgroup(own_cgroup(), unit)
    if value is not None and actual != Path(value):
        return None
    return actual


def _report_resources(channel, stop, interval):
    last_diagnostic = None

    def report():
        nonlocal last_diagnostic
        # systemd placement may not be ready at process creation; retry discovery
        # on every tick instead of freezing a missing/shared group forever.
        snapshot = _resource_snapshot(_resource_cgroup())
        diagnostic = snapshot["host_usage_error"]
        if diagnostic and diagnostic != last_diagnostic:
            print(
                f"[expman] host memory: {diagnostic}; "
                f"source={snapshot['host_usage_source']}, "
                f"limit_verified={snapshot['host_limit_verified']}",
                file=sys.stderr,
                flush=True,
            )
        last_diagnostic = diagnostic
        try:
            channel.send({"type": "resource", **snapshot})
        except OSError:
            stop.set()

    report()
    while not stop.wait(interval):
        report()
    report()


def _watch_parent():
    # EOF also covers a scheduler killed without running its finally block.
    while os.read(0, 1):
        pass
    os.killpg(os.getpgrp(), signal.SIGKILL)


def main():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    threading.Thread(target=_watch_parent, daemon=True).start()
    root = Path(sys.argv[1])
    channel = worker_channel(os.environ["EXPMAN_IPC_FD"])
    try:
        server_pid = _mps.verify_client()
    except Exception as error:
        print(f"[expman] MPS startup: {error}", file=sys.stderr, flush=True)
        channel.send(
            {
                "type": "startup_failed",
                "error_type": type(error).__qualname__,
                "error_message": str(error),
            }
        )
        return
    channel.send({"type": "mps_ready", "server": server_pid, "pid": os.getpid()})
    resource_stopped = threading.Event()
    resource_thread = threading.Thread(
        target=_report_resources,
        args=(channel, resource_stopped, float(os.environ["EXPMAN_POLL_INTERVAL"])),
        daemon=True,
    )
    resource_thread.start()
    metadata = read_record(root / "bootstrap.pkl", PickleSerializer())
    sys.path[:] = metadata["sys_path"]
    script = metadata["main_script"]
    if script is not None:
        module = types.ModuleType("__main__")
        sys.modules["__expman_main__"] = module
        namespace = runpy.run_path(script, run_name="__expman_main__")
        module.__dict__.update(namespace)
        for value in namespace.values():
            if getattr(value, "__module__", None) == "__expman_main__":
                with suppress(AttributeError, TypeError):
                    value.__module__ = "__main__"
        sys.modules["__main__"] = module
    payload = read_record(root / "input.pkl", PickleSerializer())
    experiment = Experiment(
        payload["pipeline"],
        payload["cfg"],
        run_id=payload["run_id"],
        output_dir=payload["output_dir"],
        serializer=payload["serializer"],
        _metrics_path=payload["metrics_path"],
        _cache_root=payload["cache_root"],
        _resume=True,
    )
    experiment._attempts = payload["attempts"]
    recorder = IpcRecorder(channel)
    try:
        with RunLock(experiment.output_dir):
            stage_index = payload["stage_index"]
            try:
                result = experiment.run_stage(
                    stage_index, attempt=payload["stage_attempt"], recorder=recorder
                )
            except BaseException as error:
                result = StageResult(
                    experiment.run_id,
                    stage_index,
                    payload["stage_attempt"],
                    Status.FAILED if isinstance(error, Exception) else Status.CANCELLED,
                    0.0,
                    type(error).__qualname__,
                    _error_message(error),
                )
            completed = (
                None
                if result.status.value != "succeeded"
                else experiment._store.summary((stage_index,))
            )
            with suppress(OSError):
                channel.send(
                    {
                        "type": "finished",
                        "result": result,
                        "stage": stage_index,
                        "dependencies": None
                        if completed is None
                        else completed.get("config_dependencies"),
                        "reused": bool(
                            completed is not None
                            and completed.get("_shared_reused", False)
                        ),
                    }
                )
    finally:
        resource_stopped.set()
        resource_thread.join()


if __name__ == "__main__":
    main()
