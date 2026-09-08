"""Isolated experiment interpreter; launched only by the GPU scheduler."""

import os
import pickle
import runpy
import signal
import sys
import threading
import types
from contextlib import suppress
from pathlib import Path

from .experiment import Experiment
from .storage import PickleSerializer, RunLock, read_record, write_record


class EventFile:
    def __init__(self, path):
        self.stream = path.open("ab", buffering=0)

    def record(self, event):
        payload = pickle.dumps(event, protocol=pickle.HIGHEST_PROTOCOL)
        self.stream.write(len(payload).to_bytes(8, "big") + payload)


def _watch_parent():
    # EOF also covers a scheduler killed without running its finally block.
    while os.read(0, 1):
        pass
    os.killpg(os.getpgrp(), signal.SIGKILL)


def main():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    threading.Thread(target=_watch_parent, daemon=True).start()
    root = Path(sys.argv[1])
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
        _resume=True,
    )
    experiment._attempts = payload["attempts"]
    recorder = EventFile(root / "events.bin")
    try:
        with RunLock(experiment.output_dir):
            try:
                result = experiment.run(recorder=recorder)
            except BaseException:
                result = experiment.result.attempts[-1]
            write_record(root / "result.pkl", result, experiment._store.serializer)
    finally:
        recorder.stream.close()


if __name__ == "__main__":
    main()
