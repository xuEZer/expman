import os
import sqlite3
import subprocess
import tempfile
import threading
import unittest
from itertools import pairwise
from pathlib import Path
from time import monotonic, sleep
from unittest.mock import patch

from expman import Batch, ConfigError, Experiment, Pipeline, Stage, Status, devices
from expman.devices import (
    HOST_PEAK_KB_DEFAULT,
    HOST_RESERVE_KB,
    DeviceMemory,
    HostMemory,
    MeminfoMonitor,
    MemoryObservationError,
    NvidiaMemory,
    fits_reserve,
    nvidia_smi,
)
from expman.scheduling import Gate, GpuScheduler, HostGate

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


class Prepare(Stage):
    def process(self, data, ctx):
        root = Path(ctx.cfg["markers"])
        (root / f"{ctx.run_id}.prepare").write_text(str(os.getpid()))
        return ctx.cfg["item"]


class Consume(Stage):
    def process(self, data, ctx):
        root = Path(ctx.cfg["markers"])
        (root / f"{ctx.run_id}.consume").write_text(str(os.getpid()))
        return data + ctx.cfg["offset"]


class Memory:
    def __init__(self, devices):
        self.devices = devices

    def sample(self):
        return {
            device: DeviceMemory(f"GPU-test-{device}", 16 * 1024, 15 * 1024)
            for device in self.devices
        }


class Host:
    def sample(self):
        # Roomy in bytes, with no swap in use: the reading admits launches.
        return HostMemory(64 * 1024**2, 48 * 1024**2, 0, 0)


@unittest.skipUnless(os.name == "posix", "POSIX worker process groups")
class GpuSchedulingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for target, value in [
            ("expman.devices.NvidiaMemory", Memory),
            ("expman.devices.MeminfoMonitor", Host),
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
        self.assertTrue(all(item["peak_kb"] > 0 for item in batch._stage_history))
        self.assertEqual(batch._host_memory["available_ratio"], 0.75)
        self.assertEqual(batch._host_memory["swap_free_kb"], 0.0)
        self.assertFalse(batch._host_memory["tight"])
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

    def test_top_level_stages_run_independently_and_record_dependencies(self):
        path = self.root / "two-stages.yaml"
        path.write_text(
            f"device: [0]\nmarkers: {self.root}\noffset: 10\nitem: !choice [1, 2]\n"
        )
        batch = Batch(
            Pipeline([Prepare, Consume]), path, output_dir=self.root / "two-stages"
        )

        results = batch.run(progress=False)

        self.assertEqual([result.output for result in results], [11, 12])
        self.assertEqual(
            [entry["stage"] for entry in batch._stage_history], [0, 0, 1, 1]
        )
        self.assertTrue(
            all((self.root / f"{result.run_id}.prepare").exists() for result in results)
        )
        self.assertTrue(
            all((self.root / f"{result.run_id}.consume").exists() for result in results)
        )
        dependencies = {
            entry["stage"]: {path for path, _digest in entry["dependencies"]}
            for entry in batch._stage_history
        }
        self.assertEqual(dependencies[0], {("item",), ("markers",)})
        self.assertEqual(dependencies[1], {("markers",), ("offset",)})
        scheduler = GpuScheduler(batch)
        self.assertEqual(
            scheduler._stage_paths(1), {("item",), ("markers",), ("offset",)}
        )
        self.assertGreater(scheduler._stage_duration(results[0].run_id, 1), 0)

    def test_completed_zero_gpu_sample_turns_a_stage_into_a_cpu_stage(self):
        batch = self.make_batch(count=2, devices=[0])
        scheduler = GpuScheduler(batch)
        batch._stage_history.append(
            {
                "run_id": batch.experiments[0].run_id,
                "stage": 0,
                "gpu_peak_kb": 0.0,
                "status": Status.SUCCEEDED.value,
                "capped": False,
                "gpu_capped": False,
                "dependencies": [],
            }
        )

        self.assertFalse(scheduler._stage_uses_gpu(batch.experiments[1].run_id, 0))

    def test_shared_prefix_is_loaded_by_a_new_worker(self):
        # Seed the shared cache, then let the worker processes start from it instead
        # of running the stage: both results are the seeding process, and both
        # experiments point at the same cache entry.
        output_dir = self.root / "shared-two"
        path = self.root / "shared.yaml"
        path.write_text("device: [0]\nunused: !choice [1, 2]\n")
        batch = Batch(Pipeline([SharedWork]), path, output_dir=output_dir)
        seeded = Experiment(
            Pipeline([SharedWork]),
            {"unused": 0},
            output_dir=self.root / "seed",
            _cache_root=output_dir / "cache",
        ).run()
        results = batch.run(progress=False)
        self.assertTrue(all(item.status is Status.SUCCEEDED for item in results))
        self.assertEqual(
            [item.output for item in results], [seeded.output, seeded.output]
        )
        references = [
            experiment._store.completed_reference((0,))
            for experiment in batch.experiments
        ]
        self.assertEqual(references[0], references[1])
        self.assertEqual(
            batch.experiments[1]._store.completed((0,))["state"]["producer"],
            seeded.output,
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

    def test_worker_results_are_read_back_instead_of_retained(self):
        batch = self.make_batch(count=1, devices=[0], delay=0.05)
        result = batch.run(progress=False)[0]
        attempt = result.attempts[-1]
        self.assertIsNone(attempt._output)
        self.assertIsNotNone(attempt.output_source)
        self.assertEqual(attempt.output["device"], "GPU-test-0")
        self.assertEqual(result.output["device"], "GPU-test-0")

    def test_cuda_oom_block_allows_only_existing_work_until_normal_exit(self):
        gate = Gate()
        self.assertTrue(gate.can_launch(0.005, ["running"]))
        gate.gpu_block = True
        self.assertFalse(gate.can_launch(0.99, ["running"]))
        self.assertFalse(gate.can_launch(0.99, []))

    def test_low_vram_ratio_alone_does_not_shed_attempts(self):
        batch = self.make_batch(count=1, devices=[0], delay=0)
        scheduler = GpuScheduler(batch)
        scheduler.gates[0].gpu_block = False
        memory = {0: DeviceMemory("GPU-test-0", 1000, 5)}
        host = HostMemory(64 * 1024**2, 48 * 1024**2, 0, 0)
        scheduler._relieve(memory, host)
        self.assertFalse(scheduler.gates[0].gpu_block)

    def test_host_memory_shortage_pauses_launches_until_it_recovers(self):
        batch = self.make_batch(count=1, devices=[0], delay=0.05)
        original = Host.sample
        pressured = []

        def shortage(monitor):
            if len(pressured) < 5:
                pressured.append(len(batch._active_gpu))
                return HostMemory(1000, 5, 0, 0)
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
                return HostMemory(1000, 5, 0, 0)
            if dropped and not (self.root / "release").exists():
                held.append(len(batch._active_gpu))
                if len(held) >= 5:
                    (self.root / "release").touch()
            return original(monitor)

        with (
            patch.object(devices, "HOST_RESERVE_KB", 2 * 1024**2),
            patch.object(Host, "sample", pressure),
        ):
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

    def test_host_reserve_pauses_launches_until_the_next_attempt_fits(self):
        batch = self.make_batch(count=1, devices=[0], delay=0.05)
        original = Host.sample
        pressured = []

        def scarce(monitor):
            if len(pressured) < 5:
                pressured.append(len(batch._active_gpu))
                # Room above the reserve, and not tight, but nothing like enough
                # for the peak one unknown attempt is charged.
                return HostMemory(
                    64 * 1024**2, HOST_RESERVE_KB + HOST_PEAK_KB_DEFAULT // 2, 0, 0
                )
            return original(monitor)

        with patch.object(Host, "sample", scarce):
            results = batch.run(progress=False)
        self.assertEqual(pressured, [0] * 5)
        # The refused dispatch went back to the queue instead of being lost.
        self.assertEqual(results[0].status, Status.SUCCEEDED)

    def test_observed_peak_is_recorded_and_changes_later_admission(self):
        batch = self.make_batch(count=1, devices=[0], delay=0.05)
        batch.run(progress=False)
        peaks = [item["peak_kb"] for item in batch._gpu_history]
        self.assertEqual(len(peaks), 1)
        self.assertGreaterEqual(peaks[0], 0)
        scheduler = GpuScheduler(batch)
        self.assertEqual(scheduler.peak_ceiling, peaks[0])
        self.assertEqual(scheduler._expected_peak("never-ran"), HOST_PEAK_KB_DEFAULT)

    def test_failed_host_query_sheds_newest_attempt_and_pauses_launches(self):
        batch = self.make_batch(count=2, devices=[0], delay=0.05)
        original = Host.sample
        original_launch = GpuScheduler._launch
        launches = []
        failures = 0
        launches_at_failure = []

        def counted(self, device, memory, host):
            launches.append(device)
            return original_launch(self, device, memory, host)

        def failing(monitor):
            nonlocal failures
            if failures < 2 and batch._active_gpu:
                failures += 1
                launches_at_failure.append(len(launches))
                raise MemoryObservationError("meminfo unavailable")
            if launches_at_failure and len(launches) != launches_at_failure[-1]:
                self.fail("a failed query launched work before the next reading")
            if launches_at_failure:
                launches_at_failure.pop()
            return original(monitor)

        with (
            patch.object(Host, "sample", failing),
            patch.object(GpuScheduler, "_launch", counted),
            self.assertWarns(RuntimeWarning),
        ):
            results = batch.run(progress=False)
        self.assertEqual(failures, 2)
        self.assertEqual(launches_at_failure, [])
        shed = [
            attempt
            for result in results
            for attempt in result.attempts
            if attempt.error_type == "MemoryPressure"
        ]
        self.assertEqual(len(shed), 2)
        self.assertTrue(all(item.status is Status.SUCCEEDED for item in results))

    def test_failed_memory_query_pauses_launches_without_stopping_the_batch(self):
        batch = self.make_batch(count=2, devices=[0, 1], delay=0.05)
        original = Memory.sample
        original_launch = GpuScheduler._launch
        launches = []
        failures = 0

        def counted(self, device, memory, host):
            launches.append(device)
            return original_launch(self, device, memory, host)

        def failing(monitor):
            nonlocal failures
            if failures < 3:
                failures += 1
                self.assertEqual(launches, [])
                raise MemoryObservationError("nvidia-smi unavailable")
            return original(monitor)

        with (
            patch.object(Memory, "sample", failing),
            patch.object(GpuScheduler, "_launch", counted),
            self.assertWarns(RuntimeWarning),
        ):
            results = batch.run(progress=False)
        self.assertEqual(failures, 3)
        self.assertEqual(sorted(launches), [0, 1])
        self.assertTrue(all(item.status is Status.SUCCEEDED for item in results))
        self.assertTrue(all(len(item.attempts) == 1 for item in results))
        self.assertFalse(batch._active_gpu)

    def test_memory_is_rechecked_on_every_tick(self):
        batch = self.make_batch(count=1, devices=[0], delay=0.3)
        original = Memory.sample
        stamps = []

        def recorded(monitor):
            stamps.append(monotonic())
            return original(monitor)

        with patch.object(Memory, "sample", recorded):
            batch.run(progress=False)
        gaps = [later - earlier for earlier, later in pairwise(stamps)]
        self.assertGreaterEqual(len(stamps), 10)
        self.assertLess(max(gaps), 0.1)

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

    def test_gate_ignores_ratio_and_empty_card_fallback(self):
        gate = Gate()
        self.assertTrue(gate.can_launch(0.01, ["a"]))
        self.assertTrue(gate.can_launch(0.009, ["a"]))
        gate.gpu_block = True
        self.assertFalse(gate.can_launch(0.9, ["a"]))
        self.assertFalse(gate.can_launch(0.9, []))

    def test_host_gate_holds_refills_until_a_card_is_free(self):
        gate = HostGate()
        roomy = HostMemory(64 * 1024**2, 48 * 1024**2, 0, 0)
        self.assertTrue(gate.can_launch(roomy, ["a"]))
        gate.mem_block = True
        self.assertFalse(gate.can_launch(roomy, ["a"]))
        self.assertTrue(gate.can_launch(roomy, []))
        self.assertFalse(gate.can_launch(None, []))

    def test_host_gate_refuses_room_below_the_byte_reserve(self):
        # The default reserve is 1 GiB, so the host gate refuses a reading below
        # it and admits a positive amount above it.
        gate = HostGate()
        nearly_full = HostMemory(HOST_RESERVE_KB, HOST_RESERVE_KB - 1, 0, 0)
        self.assertGreater(nearly_full.available_ratio, 0.9)
        self.assertFalse(gate.can_launch(nearly_full, []))
        roomy = HostMemory(64 * 1024**2, HOST_RESERVE_KB + 1, 0, 0)
        self.assertTrue(gate.can_launch(roomy, []))

    def test_host_gate_refuses_while_the_kernel_is_paging(self):
        # Paging means the kernel is short of RAM, whatever MemAvailable reports.
        gate = HostGate()
        large = 64 * 1024**2
        paging = HostMemory(large, large // 2, 4096, 2048, True)
        self.assertFalse(gate.can_launch(paging, []))
        # Occupancy that nothing is paging on any more admits launches again:
        # swapped pages stay out, so occupancy is a latch rather than a reading.
        settled = HostMemory(large, large // 2, 4096, 2048, False)
        self.assertTrue(gate.can_launch(settled, []))

    def test_fits_reserve_needs_both_headroom_and_no_paging(self):
        available = HOST_RESERVE_KB + 2 * 1024**2
        roomy = HostMemory(64 * 1024**2, available, 0, 0)
        paging = HostMemory(64 * 1024**2, available, 0, 0, True)
        self.assertTrue(fits_reserve(roomy, 2 * 1024**2))
        self.assertFalse(fits_reserve(roomy, 2 * 1024**2 + 1))
        self.assertFalse(fits_reserve(paging, 1))
        self.assertFalse(fits_reserve(None, 1))

    def test_query_timeout_is_an_observation_error_with_the_default_timeout(self):
        with (
            patch(
                "expman.devices.subprocess.run",
                side_effect=subprocess.TimeoutExpired("nvidia-smi", 3),
            ) as query,
            self.assertRaises(MemoryObservationError),
        ):
            NvidiaMemory((0,)).sample()
        self.assertEqual(query.call_args.kwargs["timeout"], 3.0)

    def test_nvidia_smi_is_found_outside_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            fallback = Path(temporary) / "nvidia-smi"
            fallback.write_text("#!/bin/sh\n")
            with (
                patch("expman.devices.shutil.which", return_value=None),
                patch("expman.devices.NVIDIA_SMI_FALLBACKS", (fallback,)),
                patch("expman.devices.subprocess.run") as query,
            ):
                query.return_value.stdout = "0, GPU-first, 100, 10\n"
                NvidiaMemory((0,)).sample()
            self.assertEqual(query.call_args.args[0][0], str(fallback))
        with patch("expman.devices.shutil.which", return_value="/usr/bin/nvidia-smi"):
            self.assertEqual(nvidia_smi(), "/usr/bin/nvidia-smi")

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

    def test_host_memory_reports_paging_rather_than_swap_occupancy(self):
        with tempfile.TemporaryDirectory() as temporary:
            meminfo = Path(temporary) / "meminfo"
            vmstat = Path(temporary) / "vmstat"
            meminfo.write_text(
                "MemTotal:       16777216 kB\n"
                "MemAvailable:    8388608 kB\n"
                "SwapTotal:       4194304 kB\n"
                "SwapFree:        2097152 kB\n"
            )
            vmstat.write_text("pswpin 4\npswpout 443\n")
            monitor = MeminfoMonitor(meminfo, vmstat)
            self.assertFalse(monitor.sample().paging)
            occupied = monitor.sample()
            # Half of swap is occupied while nothing is paging on it any more.
            self.assertEqual(occupied.swap_free_kb, 2097152.0)
            self.assertFalse(occupied.paging)
            self.assertFalse(occupied.tight)
            vmstat.write_text("pswpin 4\npswpout 448\n")
            paging = monitor.sample()
            self.assertTrue(paging.paging)
            self.assertTrue(paging.tight)
            vmstat.write_text("pswpin 4\npswpout 448\n")
            self.assertFalse(monitor.sample().paging)

    def test_missing_vmstat_is_not_paging_pressure(self):
        with tempfile.TemporaryDirectory() as temporary:
            meminfo = Path(temporary) / "meminfo"
            meminfo.write_text(
                "MemTotal:       16777216 kB\n"
                "MemAvailable:    8388608 kB\n"
                "SwapTotal:       4194304 kB\n"
                "SwapFree:        2097152 kB\n"
            )
            absent = Path(temporary) / "absent"
            observation = MeminfoMonitor(meminfo, absent).sample()
            self.assertFalse(observation.paging)
            self.assertFalse(observation.tight)

    @unittest.skipUnless(Path("/proc/meminfo").exists(), "host meminfo is unavailable")
    def test_default_host_monitor_reads_the_running_host(self):
        observation = MeminfoMonitor().sample()
        self.assertGreater(observation.total_kb, 0)
        self.assertTrue(0 <= observation.available_ratio <= 1)


if __name__ == "__main__":
    unittest.main()
