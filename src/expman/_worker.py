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

from .context import _error_message
from .events import Status
from .experiment import Experiment, StageResult
from .gpu_memory import limit as limit_gpu_memory
from .gpu_memory import sample as gpu_memory
from .ipc import worker_channel
from .limits import events, own_cgroup, usage
from .storage import PickleSerializer, RunLock, read_record


class IpcRecorder:
    def __init__(self, channel):
        self.channel = channel

    def record(self, event):
        self.channel.send({"type": "event", "event": event})


def _resource_snapshot(directory):
    host = usage(directory)
    return {
        "timestamp": monotonic(),
        "host_current_kb": None if host is None else host[0],
        "host_peak_kb": None if host is None else host[1],
        "gpu": gpu_memory(),
        "memory_events": events(directory),
        "cgroup": None if directory is None else str(directory),
    }


def _resource_cgroup():
    """Use a direct-cgroup path, or discover the systemd scope from within it."""
    if "EXPMAN_CGROUP" in os.environ:
        return Path(os.environ["EXPMAN_CGROUP"])
    return own_cgroup()


def _report_resources(channel, stop, interval):
    directory = _resource_cgroup()

    def report():
        try:
            channel.send({"type": "resource", **_resource_snapshot(directory)})
        except OSError:
            stop.set()

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
    resource_stopped = threading.Event()
    resource_thread = threading.Thread(
        target=_report_resources,
        args=(channel, resource_stopped, float(os.environ["EXPMAN_POLL_INTERVAL"])),
        daemon=True,
    )
    resource_thread.start()
    if "EXPMAN_GPU_LIMIT_KB" in os.environ:
        with suppress(ValueError):
            limit_gpu_memory(float(os.environ["EXPMAN_GPU_LIMIT_KB"]))
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
            try:
                stage_index = payload.get("stage_index")
                result = (
                    experiment.run(recorder=recorder)
                    if stage_index is None
                    else experiment.run_stage(
                        stage_index, attempt=payload["stage_attempt"], recorder=recorder
                    )
                )
            except BaseException as error:
                if stage_index is None:
                    result = experiment.result.attempts[-1]
                else:
                    result = StageResult(
                        experiment.run_id,
                        stage_index,
                        payload["stage_attempt"],
                        Status.FAILED
                        if isinstance(error, Exception)
                        else Status.CANCELLED,
                        0.0,
                        type(error).__qualname__,
                        _error_message(error),
                    )
            completed = (
                None
                if stage_index is None or result.status.value != "succeeded"
                else experiment._store.completed((stage_index,))
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
                        "resources": _resource_snapshot(_resource_cgroup()),
                    }
                )
    finally:
        resource_stopped.set()
        resource_thread.join()


if __name__ == "__main__":
    main()
