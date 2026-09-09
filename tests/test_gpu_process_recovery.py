import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from time import monotonic, sleep

SCRIPT = """
import os
import random
import sys
from pathlib import Path
from time import sleep
from expman import Batch, Pipeline, Stage
from expman import devices

class Memory:
    def __init__(self, selected):
        self.selected = selected
    def sample(self):
        return {index: devices.DeviceMemory(f"GPU-test-{index}", 100, 90)
                for index in self.selected}

class Work(Stage):
    def process(self, data, ctx):
        root = Path(ctx.cfg["markers"])
        ctx.state.setdefault("draws", []).append(random.random())
        ctx.state["count"] = ctx.state.get("count", 0) + 1
        ctx.checkpoint.save()
        (root / (ctx.run_id + ".pid")).write_text(str(os.getpid()))
        if ctx.attempt == 1:
            try:
                sleep(60)
            finally:
                (root / (ctx.run_id + ".finally")).touch()
        return ctx.state

if __name__ == "__main__":
    root = Path(sys.argv[1])
    devices.NvidiaMemory = Memory
    devices.LAUNCH_INTERVAL = 0.02
    devices.POLL_INTERVAL = 0.01
    if len(sys.argv) > 2:
        batch = Batch.resume(Pipeline([Work]), root / "batch")
    else:
        cfg = root / "experiment.yaml"
        cfg.write_text(f"device: [0, 1]\\nmarkers: {root}\\nitem: !choice [1, 2]\\n")
        batch = Batch(Pipeline([Work]), cfg, output_dir=root / "batch")
    results = batch.run(refresh_interval=0.02)
    expected = random.Random(0)
    values = [expected.random(), expected.random()]
    assert all(result.output == {"count": 2, "draws": values} for result in results), results
    assert all(len(result.attempts) == 2 for result in results), results
"""


def alive(pid):
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists() and stat.read_text().split(")", 1)[1].split()[0] == "Z":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@unittest.skipUnless(sys.platform == "linux", "Linux process lifecycle")
class ProcessRecoveryTests(unittest.TestCase):
    def test_real_sigint_and_parent_kill_cleanup_and_resume(self):
        for interruption in (signal.SIGINT, signal.SIGKILL):
            with (
                self.subTest(signal=interruption),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                script = root / "run.py"
                script.write_text(SCRIPT)
                with (root / "parent.log").open("wb") as log:
                    process = subprocess.Popen(
                        [sys.executable, str(script), str(root)],
                        stdout=log,
                        stderr=log,
                        start_new_session=True,
                    )
                    pids = []
                    try:
                        deadline = monotonic() + 10
                        while len(list(root.glob("*.pid"))) < 2:
                            if process.poll() is not None or monotonic() > deadline:
                                self.fail((root / "parent.log").read_text())
                            sleep(0.02)
                        pids = [int(path.read_text()) for path in root.glob("*.pid")]
                        process.send_signal(interruption)
                        process.wait(timeout=10)
                        deadline = monotonic() + 5
                        while (
                            any(alive(pid) for pid in pids) and monotonic() < deadline
                        ):
                            sleep(0.02)
                        self.assertFalse(any(alive(pid) for pid in pids))
                        self.assertFalse(list(root.glob("*.finally")))
                        result = subprocess.run(
                            [sys.executable, str(script), str(root), "resume"],
                            capture_output=True,
                            text=True,
                            timeout=15,
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn("已结束 2/2 (100%)", result.stderr)
                        self.assertIn("GPU0:", result.stderr)
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait()
                        for pid in pids:
                            if alive(pid):
                                os.killpg(pid, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
