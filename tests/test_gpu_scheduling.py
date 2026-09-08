import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from time import monotonic, sleep
from unittest.mock import patch

from expman import Batch, ConfigError, Pipeline, Stage, Status
from expman.devices import DeviceMemory, NvidiaMemory
from expman.scheduling import Gate

IMPORT_DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES")


class GpuWork(Stage):
    def process(self, data, ctx):
        root = Path(ctx.cfg["markers"])
        (root / f"{ctx.run_id}.started").write_text(str(os.getpid()))
        ctx.state["saved"] = ctx.state.get("saved", 0) + 1
        ctx.checkpoint.save()
        (root / f"{ctx.run_id}.checkpoint").touch()
        if ctx.cfg.get("crash"):
            os._exit(7)
        if ctx.cfg.get("fail_once") and ctx.attempt == 1:
            raise ValueError("retry me")
        for step in range(10):
            ctx.log_metrics({"train": {"loss": step}}, step=step)
        try:
            if ctx.attempt == 1:
                deadline = monotonic() + 5
                while (
                    ctx.cfg.get("wait_for_release") and not (root / "release").exists()
                ):
                    if monotonic() > deadline:
                        raise RuntimeError("test release timed out")
                    sleep(0.01)
                sleep(ctx.cfg.get("delay", 0.1))
            (root / f"{ctx.run_id}.finished").touch()
            return {
                "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "import_device": IMPORT_DEVICE,
                "pid": os.getpid(),
                "saved": ctx.state["saved"],
                "cfg_devices": list(ctx.cfg["device"]),
            }
        finally:
            (root / f"{ctx.run_id}.finally").touch()


class Memory:
    def __init__(self, devices):
        self.devices = devices

    def sample(self):
        return {
            device: DeviceMemory(f"GPU-test-{device}", 100, 90)
            for device in self.devices
        }


@unittest.skipUnless(os.name == "posix", "POSIX worker process groups")
class GpuSchedulingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for target, value in [
            ("expman.devices.NvidiaMemory", Memory),
            ("expman.devices.LAUNCH_INTERVAL", 0.02),
            ("expman.devices.POLL_INTERVAL", 0.01),
        ]:
            mock = patch(target, value)
            mock.start()
            self.addCleanup(mock.stop)

    def make_batch(self, *, count=3, devices=None, **options):
        import yaml

        cfg = {
            "device": [0, 1] if devices is None else devices,
            "markers": str(self.root),
            **options,
        }
        path = self.root / "experiments.yaml"
        path.write_text(yaml.safe_dump(cfg) + f"item: !choice {list(range(count))}\n")
        return Batch(Pipeline([GpuWork]), path, output_dir=self.root / "batch")

    def test_multiple_devices_and_metrics_results_are_collected(self):
        batch = self.make_batch(count=4, delay=0.25)
        results = batch.run(progress=False)
        self.assertTrue(all(item.status is Status.SUCCEEDED for item in results))
        self.assertEqual(
            {item.output["device"] for item in results}, {"GPU-test-0", "GPU-test-1"}
        )
        self.assertEqual(len({item.output["pid"] for item in results}), 4)
        self.assertTrue(
            all(
                item.output["import_device"] == item.output["device"]
                for item in results
            )
        )
        self.assertTrue(all(item.output["cfg_devices"] == [0, 1] for item in results))
        self.assertTrue(any(item["concurrency"] > 1 for item in batch._gpu_history))
        with sqlite3.connect(batch.output_dir / "metrics.sqlite3") as db:
            self.assertEqual(
                db.execute("SELECT count(*) FROM metrics").fetchone()[0], 40
            )
        self.assertTrue(batch.recorder.events)
        resumed = Batch.resume(Pipeline([GpuWork]), batch.output_dir)
        self.assertEqual(
            [item.output for item in resumed.run(progress=False)],
            [item.output for item in results],
        )
        self.assertEqual(resumed.devices, (0, 1))

    def test_failure_and_unexpected_process_exit_get_one_retry(self):
        for crash in (False, True):
            with self.subTest(crash=crash), tempfile.TemporaryDirectory() as temporary:
                batch = Batch(
                    Pipeline([GpuWork]),
                    {
                        "device": [0],
                        "markers": temporary,
                        "crash": crash,
                        "fail_once": not crash,
                    },
                    output_dir=Path(temporary) / "batch",
                )
                result = batch.run(progress=False)[0]
                self.assertEqual(len(result.attempts), 2)
                self.assertEqual(
                    result.status, Status.FAILED if crash else Status.SUCCEEDED
                )
                self.assertEqual(
                    result.attempts[0].error_type,
                    "WorkerExit" if crash else "ValueError",
                )

    def test_interrupt_kills_all_without_finally_and_restores_checkpoint(self):
        batch = self.make_batch(count=2, delay=30)
        original = Memory.sample

        def interrupt(monitor):
            if len(list(self.root.glob("*.checkpoint"))) == 2:
                raise KeyboardInterrupt
            return original(monitor)

        with (
            patch.object(Memory, "sample", interrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            batch.run(progress=False)
        self.assertEqual(len(list(self.root.glob("*.finally"))), 0)
        for result in batch.results:
            self.assertEqual(result.status, Status.CANCELLED)
            pid = int((self.root / f"{result.run_id}.started").read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
        resumed = Batch.resume(Pipeline([GpuWork]), batch.output_dir)
        results = resumed.run(progress=False)
        self.assertTrue(all(item.status is Status.SUCCEEDED for item in results))
        self.assertTrue(all(item.output["saved"] == 2 for item in results))
        self.assertTrue(all(len(item.attempts) == 2 for item in results))

    def test_memory_kill_waits_for_other_exit_then_retries(self):
        batch = self.make_batch(count=2, devices=[0], delay=0, wait_for_release=True)
        dropped = []
        observed_block = []
        original = Memory.sample

        def pressure(monitor):
            active = list(batch._active_gpu)
            if (
                not dropped
                and len(active) == 2
                and all((self.root / f"{key}.checkpoint").exists() for key in active)
            ):
                dropped.append(active[-1])
                return {0: DeviceMemory("GPU-test-0", 100, 9)}
            if dropped and not list(self.root.glob("*.finished")):
                observed_block.append(tuple(active))
                self.assertEqual(len(active), 1)
                if len(observed_block) >= 5:
                    (self.root / "release").touch()
            return original(monitor)

        with patch.object(Memory, "sample", pressure):
            batch.run(progress=False)
        self.assertTrue(dropped)
        self.assertTrue(observed_block)
        result = next(item for item in batch.results if item.run_id == dropped[0])
        self.assertEqual(result.attempts[0].status, Status.CANCELLED)
        self.assertEqual(result.attempts[0].error_type, "MemoryPressure")
        self.assertEqual(result.status, Status.SUCCEEDED)
        self.assertEqual(result.output["saved"], 2)

    def test_missing_memory_observation_stops_and_cleans_up(self):
        batch = self.make_batch(count=1, delay=30)
        original = Memory.sample

        def broken(monitor):
            if list(self.root.glob("*.checkpoint")):
                raise RuntimeError("monitor unavailable")
            return original(monitor)

        with (
            patch.object(Memory, "sample", broken),
            self.assertRaisesRegex(RuntimeError, "monitor unavailable"),
        ):
            batch.run(progress=False)
        self.assertEqual(batch.results[0].status, Status.CANCELLED)
        self.assertFalse(batch._active_gpu)


class DeviceTests(unittest.TestCase):
    def test_device_is_a_batch_wide_plain_list(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for value in (None, 0, [True], [-1], [0, 0], ["0"]):
                with self.subTest(value=value), self.assertRaises(ConfigError):
                    Batch(Pipeline([]), {"device": value}, output_dir=root / "unused")
            cfg = root / "experiment.yaml"
            cfg.write_text("device: !choice [[0], [1]]\n")
            with self.assertRaises(ConfigError):
                Batch(Pipeline([]), cfg, output_dir=root / "unused")
            cfg.write_text("device: [0, 1]\nx: !choice [1, 2]\n")
            batch = Batch(Pipeline([]), cfg, output_dir=root / "batch")
            self.assertEqual(len(batch.experiments), 2)
            self.assertEqual(batch.devices, (0, 1))

    def test_gate_boundary_spacing_and_empty_card_fallback(self):
        gate = Gate(last_launch=10)
        self.assertTrue(gate.can_launch(0.1, 15, ["a"]))
        self.assertFalse(gate.can_launch(0.099, 15, ["a"]))
        self.assertFalse(gate.can_launch(0.5, 14.99, ["a"]))
        gate.blocked = True
        self.assertFalse(gate.can_launch(0.9, 100, ["a"]))
        self.assertTrue(gate.can_launch(0.9, 100, []))

    def test_nvidia_memory_validates_observations_and_identities(self):
        with patch("expman.devices.subprocess.run") as query:
            query.return_value.stdout = (
                "0, GPU-first, 100, 10\n1, GPU-second, 200, 150\n"
            )
            monitor = NvidiaMemory((0, 1))
            self.assertEqual(monitor.sample()[0].free_ratio, 0.1)
            self.assertIn("--id=0,1", query.call_args.args[0])
            query.return_value.stdout = (
                "0, GPU-replaced, 100, 10\n1, GPU-second, 200, 150\n"
            )
            with self.assertRaisesRegex(RuntimeError, "identities changed"):
                monitor.sample()
            query.return_value.stdout = "0, GPU-first, N/A, N/A\n"
            with self.assertRaises(RuntimeError):
                NvidiaMemory((0,)).sample()


if __name__ == "__main__":
    unittest.main()
