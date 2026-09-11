import os
import sqlite3
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from time import monotonic, sleep
from unittest.mock import patch

from expman import Batch, ConfigError, Pipeline, Stage, Status
from expman.devices import (
    DeviceMemory,
    HostMemory,
    MeminfoMonitor,
    MemoryObservationError,
    NvidiaMemory,
)
from expman.scheduling import Gate

IMPORT_DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES")


class GpuWork(Stage):
    def process(self, data, ctx):
        root = Path(ctx.cfg["markers"])
        (root / f"{ctx.run_id}.started").write_text(str(os.getpid()))
        ctx.state["saved"] = ctx.state.get("saved", 0) + 1
        ctx.checkpoint.save()
        (root / f"{ctx.run_id}.checkpoint").touch()
        if ctx.cfg["crash"]:
            os._exit(7)
        if ctx.cfg["fail_once"] and ctx.attempt == 1:
            raise ValueError("retry me")
        for step in range(10):
            ctx.log_metrics({"train": {"loss": step}}, step=step)
        try:
            if ctx.attempt == 1:
                deadline = monotonic() + 5
                while ctx.cfg["wait_for_release"] and not (root / "release").exists():
                    if monotonic() > deadline:
                        raise RuntimeError("test release timed out")
                    sleep(0.01)
                sleep(ctx.cfg["delay"])
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


class SharedWork(Stage):
    def process(self, data, ctx):
        ctx.state["producer"] = os.getpid()
        ctx.log_metrics({"loss": 0.25}, step=0)
        return os.getpid()


class Memory:
    def __init__(self, devices):
        self.devices = devices

    def sample(self):
        return {
            device: DeviceMemory(f"GPU-test-{device}", 100, 90)
            for device in self.devices
        }


class Host:
    def sample(self):
        return HostMemory(1000, 900, 800, 400)


@unittest.skipUnless(os.name == "posix", "POSIX worker process groups")
class GpuSchedulingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for target, value in [
            ("expman.devices.NvidiaMemory", Memory),
            ("expman.devices.MeminfoMonitor", Host),
            ("expman.devices.LAUNCH_INTERVAL", 0.02),
            ("expman.devices.POLL_INTERVAL", 0.01),
            ("expman.devices.QUERY_RETRY_INTERVAL", 0.01),
        ]:
            mock = patch(target, value)
            mock.start()
            self.addCleanup(mock.stop)

    def make_batch(self, *, count=3, devices=None, **options):
        import yaml

        cfg = {
            "device": [0, 1] if devices is None else devices,
            "markers": str(self.root),
            "crash": False,
            "fail_once": False,
            "wait_for_release": False,
            "delay": 0.1,
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
        self.assertEqual(batch._host_memory["available_ratio"], 0.9)
        self.assertEqual(batch._host_memory["swap_free_kb"], 400.0)
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

    def test_shared_prefix_is_loaded_by_a_new_worker(self):
        # Use two distinct configs whose difference is irrelevant to SharedWork.
        path = self.root / "shared.yaml"
        path.write_text("device: [0]\nunused: !choice [1, 2]\n")
        batch = Batch(Pipeline([SharedWork]), path, output_dir=self.root / "shared-two")
        with patch("expman.devices.LAUNCH_INTERVAL", 1.0):
            results = batch.run(progress=False)
        self.assertTrue(all(item.status is Status.SUCCEEDED for item in results))
        self.assertEqual(results[0].output, results[1].output)
        references = [
            experiment._store.completed_reference((0,))
            for experiment in batch.experiments
        ]
        self.assertEqual(references[0], references[1])
        self.assertEqual(
            batch.experiments[1]._store.completed((0,))["state"]["producer"],
            results[0].output,
        )

    def test_failure_and_unexpected_process_exit_get_one_retry(self):
        for crash in (False, True):
            with self.subTest(crash=crash), tempfile.TemporaryDirectory() as temporary:
                batch = Batch(
                    Pipeline([GpuWork]),
                    {
                        "device": [0],
                        "markers": temporary,
                        "wait_for_release": False,
                        "delay": 0.1,
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

    def test_host_memory_shortage_pauses_launches_until_it_recovers(self):
        batch = self.make_batch(count=1, devices=[0], delay=0.05)
        original = Host.sample
        pressured = []

        def shortage(monitor):
            if len(pressured) < 5:
                pressured.append(len(batch._active_gpu))
                return HostMemory(1000, 99, 0, 0)
            return original(monitor)

        with patch.object(Host, "sample", shortage):
            results = batch.run(progress=False)
        self.assertEqual(pressured, [0] * 5)
        self.assertEqual(results[0].status, Status.SUCCEEDED)

    def test_host_memory_pressure_sheds_newest_attempt_and_holds_other_cards(self):
        batch = self.make_batch(count=2, devices=[0, 1], delay=0, wait_for_release=True)
        dropped = []
        held = []
        cards = []
        original = Host.sample

        def pressure(monitor):
            active = list(batch._active_gpu)
            if (
                not dropped
                and len(active) == 2
                and all((self.root / f"{key}.checkpoint").exists() for key in active)
            ):
                dropped.append(active[-1])
                cards.extend(batch._active_gpu.values())
                return HostMemory(1000, 50, 0, 0)
            if dropped and not (self.root / "release").exists():
                held.append(len(batch._active_gpu))
                if len(held) >= 5:
                    (self.root / "release").touch()
            return original(monitor)

        with patch.object(Host, "sample", pressure):
            results = batch.run(progress=False)
        self.assertTrue(dropped)
        # The attempts ran on separate cards, so the shed run could have been
        # replaced straight away on the freed card if host RAM had not held it.
        self.assertEqual(sorted(cards), [0, 1])
        # While the surviving attempt runs, no replacement starts on any card.
        self.assertEqual(set(held), {1})
        result = next(item for item in results if item.run_id == dropped[0])
        self.assertEqual(result.attempts[0].status, Status.CANCELLED)
        self.assertEqual(result.attempts[0].error_type, "MemoryPressure")
        self.assertEqual(result.status, Status.SUCCEEDED)
        self.assertEqual(len(result.attempts), 2)
        self.assertTrue(all(item.status is Status.SUCCEEDED for item in results))

    def test_transient_memory_failure_keeps_worker_and_pauses_launches(self):
        batch = self.make_batch(count=2, delay=0.2, devices=[0])
        original = Memory.sample
        failures = 0
        observations_after_failure = 0

        def intermittent(monitor):
            nonlocal failures, observations_after_failure
            if batch._active_gpu and failures < 2:
                failures += 1
                self.assertEqual(len(batch._active_gpu), 1)
                raise MemoryObservationError("query timeout")
            if failures:
                observations_after_failure += 1
            return original(monitor)

        with (
            patch.object(Memory, "sample", intermittent),
            self.assertWarns(RuntimeWarning),
        ):
            results = batch.run(progress=False)
        self.assertEqual(failures, 2)
        self.assertGreater(observations_after_failure, 0)
        self.assertTrue(all(result.status is Status.SUCCEEDED for result in results))
        self.assertTrue(all(len(result.attempts) == 1 for result in results))

    def test_query_failure_counter_resets_after_success(self):
        batch = self.make_batch(count=1, delay=0.1)
        original = Memory.sample
        calls = 0

        def alternating(monitor):
            nonlocal calls
            calls += 1
            if calls in (1, 2, 4, 5):
                raise MemoryObservationError("temporary query failure")
            return original(monitor)

        with (
            patch.object(Memory, "sample", alternating),
            self.assertWarns(RuntimeWarning),
        ):
            results = batch.run(progress=False)
        self.assertEqual(results[0].status, Status.SUCCEEDED)
        self.assertEqual(len(results[0].attempts), 1)

    def test_slow_background_scores_do_not_block_worker_launches(self):
        batch = self.make_batch(count=2, delay=0.1, devices=[0])
        entered = threading.Event()
        release = threading.Event()
        errors = []

        def slow_scores(pending):
            entered.set()
            if not release.wait(20):
                raise RuntimeError("test did not release background computation")

        def run():
            try:
                batch.run(progress=False)
            except BaseException as error:
                errors.append(error)

        with patch.object(
            batch._time_estimator, "refresh_priorities", side_effect=slow_scores
        ):
            thread = threading.Thread(target=run)
            thread.start()
            try:
                self.assertTrue(entered.wait(3))
                deadline = monotonic() + 10
                while (
                    len(list(self.root.glob("*.checkpoint"))) < 2
                    and monotonic() < deadline
                ):
                    sleep(0.01)
                self.assertEqual(len(list(self.root.glob("*.checkpoint"))), 2)
                self.assertFalse(release.is_set())
            finally:
                release.set()
                thread.join(20)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(
            all(result.status is Status.SUCCEEDED for result in batch.results)
        )

    def test_missing_memory_observation_stops_and_cleans_up(self):
        batch = self.make_batch(count=1, delay=30)
        original = Memory.sample

        failures = 0

        def broken(monitor):
            nonlocal failures
            if list(self.root.glob("*.checkpoint")):
                failures += 1
                raise MemoryObservationError("monitor unavailable")
            return original(monitor)

        with (
            patch.object(Memory, "sample", broken),
            self.assertRaisesRegex(
                RuntimeError, "3 consecutive queries: monitor unavailable"
            ),
            self.assertWarns(RuntimeWarning),
        ):
            batch.run(progress=False)
        self.assertEqual(batch.results[0].status, Status.CANCELLED)
        self.assertFalse(batch._active_gpu)
        self.assertEqual(failures, 3)


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

    def test_query_timeout_is_retryable_and_uses_internal_timeout(self):
        with (
            patch(
                "expman.devices.subprocess.run",
                side_effect=subprocess.TimeoutExpired("nvidia-smi", 10),
            ) as query,
            self.assertRaises(MemoryObservationError),
        ):
            NvidiaMemory((0,)).sample()
        self.assertEqual(query.call_args.kwargs["timeout"], 10.0)

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

    def test_host_memory_reads_meminfo_and_rejects_invalid_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "meminfo"
            path.write_text(
                "MemTotal:       1000 kB\n"
                "MemFree:         100 kB\n"
                "MemAvailable:    250 kB\n"
                "SwapTotal:       400 kB\n"
                "SwapFree:        300 kB\n"
            )
            observation = MeminfoMonitor(path).sample()
            self.assertEqual(observation.available_ratio, 0.25)
            self.assertEqual(observation.swap_total_kb, 400.0)
            self.assertEqual(observation.swap_free_kb, 300.0)
            path.write_text("MemTotal: 1000 kB\nMemAvailable: 2000 kB\n")
            with self.assertRaisesRegex(MemoryObservationError, "invalid host memory"):
                MeminfoMonitor(path).sample()
            path.write_text("MemTotal: 1000 kB\nMemAvailable: N/A\n")
            with self.assertRaisesRegex(
                MemoryObservationError, "could not observe host memory"
            ):
                MeminfoMonitor(path).sample()
            path.write_text("MemFree: 100 kB\n")
            with self.assertRaisesRegex(
                MemoryObservationError, "could not observe host memory"
            ):
                MeminfoMonitor(path).sample()

    @unittest.skipUnless(Path("/proc/meminfo").exists(), "host meminfo is unavailable")
    def test_default_host_monitor_reads_the_running_host(self):
        observation = MeminfoMonitor().sample()
        self.assertGreater(observation.total_kb, 0)
        self.assertTrue(0 <= observation.available_ratio <= 1)


if __name__ == "__main__":
    unittest.main()
