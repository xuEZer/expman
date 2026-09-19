import os
import sqlite3
import subprocess
import tempfile
import unittest
from itertools import pairwise
from pathlib import Path
from time import monotonic, sleep
from unittest.mock import Mock, patch

from expman import Batch, ConfigError, Experiment, Pipeline, Stage, Status, devices
from expman.devices import (
    GPU_PEAK_KB_DEFAULT,
    HOST_RESERVE_KB,
    DeviceMemory,
    HostMemory,
    MeminfoMonitor,
    MemoryObservationError,
    NvidiaMemory,
    fits_reserve,
    nvidia_smi,
)
from expman.scheduling import GpuScheduler, HostGate

IMPORT_DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES")


class GpuWork(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {
            "markers": True,
            "item": True,
            "crash": True,
            "fail_once": True,
            "wait_for_release": True,
            "delay": True,
            "device": True,
        }

    def process(self, data, ctx):
        root = Path(ctx.cfg["markers"])
        ctx.cfg["item"]
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
    @classmethod
    def config_dependencies(cls, cfg):
        return {}

    def process(self, data, ctx):
        ctx.state["producer"] = os.getpid()
        ctx.log_metrics({"loss": 0.25}, step=0)
        return os.getpid()


class Prepare(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {"markers": True, "item": True}

    def process(self, data, ctx):
        root = Path(ctx.cfg["markers"])
        (root / f"{ctx.run_id}.prepare").write_text(str(os.getpid()))
        return ctx.cfg["item"]


class Consume(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {"markers": True, "offset": True}

    def process(self, data, ctx):
        root = Path(ctx.cfg["markers"])
        (root / f"{ctx.run_id}.consume").write_text(str(os.getpid()))
        return data + ctx.cfg["offset"]


class ConditionalWork(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        dependencies = {"Baseline": True, "imputer": True}
        if cfg["Baseline"] == "B2":
            dependencies["predictor"] = True
        return dependencies

    def process(self, data, ctx):
        return ctx.cfg["imputer"]


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


def _captured_gpu_ceilings(batch):
    """Run the Batch, returning what ceiling each attempt was started with.

    ``subprocess.run`` reaches the same ``Popen`` this patches, so only the
    worker launches are captured; every other call is forwarded untouched.
    """
    ceilings = []
    original = subprocess.Popen

    def captured(*args, **kwargs):
        argv = args[0] if args else kwargs.get("args", ())
        if "expman._worker" in argv:
            ceilings.append(kwargs.get("env", {}).get("EXPMAN_GPU_LIMIT_KB"))
        return original(*args, **kwargs)

    with patch("expman.scheduling.subprocess.Popen", side_effect=captured):
        batch.run(progress=False)
    return ceilings


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
            "seed": 0,
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

    def warm(self, batch, *, gpu_peak_kb=1024, peak_kb=4096):
        """Give every Stage the completed sample a previous run would have left.

        A Batch measures each Stage once before it starts packing attempts
        concurrently, and a Stage that has never reported CUDA usage receives no
        GPU contract afterwards. Tests about packing therefore start from a Batch
        whose Stages have already been measured, with a CUDA peak on record.
        """
        for experiment in batch.experiments:
            for stage_index in range(len(experiment.pipeline.stages)):
                batch._stage_history.append(
                    {
                        "run_id": experiment.run_id,
                        "stage": stage_index,
                        "status": Status.SUCCEEDED.value,
                        "peak_kb": float(peak_kb),
                        "gpu_peak_kb": float(gpu_peak_kb),
                        "gpu_capped": False,
                        "capped": False,
                        "reused": False,
                    }
                )
        return batch

    def test_multiple_devices_and_metrics_results_are_collected(self):
        batch = self.warm(self.make_batch(count=4, delay=0.25))
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
        measured = [item for item in batch._stage_history if not item.get("reused")]
        self.assertTrue(measured)
        self.assertTrue(all(item["peak_kb"] > 0 for item in measured))
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
            f"device: [0]\nseed: 0\nmarkers: {self.root}\noffset: 10\nitem: !choice [1, 2]\n"
        )
        batch = Batch(
            Pipeline([Prepare, Consume]), path, output_dir=self.root / "two-stages"
        )

        results = batch.run(progress=False)

        self.assertEqual([result.output for result in results], [11, 12])
        self.assertEqual(
            sorted(entry["stage"] for entry in batch._stage_history), [0, 0, 1, 1]
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

    def test_completed_zero_gpu_sample_needs_no_vram_reservation(self):
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

        self.assertEqual(
            scheduler._stage_peak(
                batch.experiments[1].run_id, 0, "gpu_peak_kb", GPU_PEAK_KB_DEFAULT
            ),
            0.0,
        )

    def test_gpu_contract_never_exceeds_physical_card_capacity(self):
        batch = self.make_batch(count=2, devices=[0])
        scheduler = GpuScheduler(batch)
        capacity_kb = 3 * 1024

        with patch.object(scheduler, "_stage_peak", return_value=100 * capacity_kb):
            candidates = scheduler._stage_candidates(0, capacity_kb)

        self.assertTrue(candidates)
        self.assertTrue(all(item.gpu_contract_kb == capacity_kb for item in candidates))

    def test_known_stage_group_has_only_one_runnable_representative(self):
        path = self.root / "groups.yaml"
        path.write_text(
            f"device: [0]\nseed: 0\nmarkers: {self.root}\nunused: !choice [1, 2, 3]\n"
        )
        batch = Batch(Pipeline([SharedWork]), path, output_dir=self.root / "groups")
        scheduler = GpuScheduler(batch)

        candidates = scheduler._stage_candidates(0, 16 * 1024**2)

        self.assertEqual(len(candidates), 1)

    def test_stage_group_follows_declared_dependencies(self):
        path = self.root / "declared.yaml"
        path.write_text(
            f"device: [0]\nseed: 0\nmarkers: {self.root}\nitem: !choice [1, 2]\n"
            "crash: false\nfail_once: false\nwait_for_release: false\ndelay: 0\n"
        )
        batch = Batch(Pipeline([GpuWork]), path, output_dir=self.root / "declared")
        scheduler = GpuScheduler(batch)
        first, second = (item.run_id for item in batch.experiments)

        # GpuWork declares item, so its two values are separate groups.
        self.assertNotEqual(
            scheduler._stage_group(first, 0), scheduler._stage_group(second, 0)
        )
        self.assertEqual(
            scheduler._stage_paths(0),
            {
                ("markers",),
                ("item",),
                ("crash",),
                ("fail_once",),
                ("wait_for_release",),
                ("delay",),
                ("device",),
            },
        )

    def test_undeclared_configuration_does_not_split_stage_groups(self):
        path = self.root / "undeclared.yaml"
        path.write_text(
            f"device: [0]\nseed: 0\nmarkers: {self.root}\nitem: 0\n"
            "crash: false\nfail_once: false\nwait_for_release: false\ndelay: 0\n"
            "unused: !choice [1, 2]\n"
        )
        batch = Batch(Pipeline([GpuWork]), path, output_dir=self.root / "undeclared")
        scheduler = GpuScheduler(batch)
        first, second = (item.run_id for item in batch.experiments)

        # unused is not declared, so the two runs share one reusable group.
        self.assertEqual(
            scheduler._stage_group(first, 0), scheduler._stage_group(second, 0)
        )

    def test_conditional_dependencies_follow_the_config_value(self):
        path = self.root / "conditional.yaml"
        path.write_text(
            "device: [0]\nseed: 0\n"
            "Baseline: !choice [B1, B2]\nimputer: i\n"
            "predictor: !choice [p1, p2]\n"
        )
        batch = Batch(
            Pipeline([ConditionalWork]), path, output_dir=self.root / "conditional"
        )
        scheduler = GpuScheduler(batch)
        groups = {
            (experiment._cfg["Baseline"], experiment._cfg["predictor"]): (
                scheduler._stage_group(experiment.run_id, 0)
            )
            for experiment in batch.experiments
        }

        # B1 declares imputer only, so predictor must not split its groups;
        # B2 declares predictor, so its runs stay distinct.
        self.assertEqual(groups[("B1", "p1")], groups[("B1", "p2")])
        self.assertNotEqual(groups[("B2", "p1")], groups[("B2", "p2")])
        self.assertNotEqual(groups[("B1", "p1")], groups[("B2", "p1")])

    def test_tick_greedily_fills_a_card(self):
        batch = self.warm(self.make_batch(count=4, devices=[0]))
        scheduler = GpuScheduler(batch)
        started = []

        def launch(device, memory, host, candidate):
            started.append(device)
            batch._queue.remove(candidate.run_id)
            return True

        with patch.object(scheduler, "_launch", side_effect=launch):
            scheduler._launch_available(Memory([0]).sample(), Host().sample())

        self.assertEqual(started, [0, 0, 0, 0])

    def test_tick_plan_combines_memory_and_gpu_stages(self):
        path = self.root / "two-stage.yaml"
        path.write_text(
            f"device: [0]\nseed: 0\nmarkers: {self.root}\n"
            "crash: false\nfail_once: false\nwait_for_release: false\ndelay: 0\n"
            "item: !choice [0, 1, 2, 3, 4, 5, 6, 7]\n"
        )
        batch = Batch(Pipeline([GpuWork, GpuWork]), path, output_dir=self.root / "plan")
        for experiment in batch.experiments[4:]:
            batch._stage_progress[experiment.run_id] = 1
        scheduler = GpuScheduler(batch)

        def peak(run_id, stage, field, default):
            if field == "peak_kb":
                return (512 if stage == 0 else 1024) * 1024
            return 0.0 if stage == 0 else 1024 * 1024

        host = HostMemory(64 * 1024**2, HOST_RESERVE_KB + 5 * 1024**2, 0, 0)
        memory = {0: DeviceMemory("GPU-test-0", 3000, 3000)}
        with patch.object(scheduler, "_stage_peak", side_effect=peak):
            plan = scheduler._launch_plan(memory, host)

        stages = [candidate.stage_index for candidate, _device in plan]
        self.assertEqual(stages.count(0), 4)
        self.assertEqual(stages.count(1), 2)

    def test_shared_prefix_is_materialized_without_worker_per_member(self):
        # SharedWork declares no dependencies, so every Batch member shares one
        # reusable group and receives a local cache reference without a worker.
        output_dir = self.root / "shared-two"
        path = self.root / "shared.yaml"
        path.write_text("device: [0]\nseed: 0\nunused: !choice [1, 2]\n")
        batch = Batch(Pipeline([SharedWork]), path, output_dir=output_dir)
        seeded = Experiment(
            Pipeline([SharedWork]),
            {"seed": 0, "unused": 0},
            output_dir=self.root / "seed",
            _cache_root=output_dir / "cache",
        ).run()
        launches = []
        original_launch = GpuScheduler._launch

        def counted(scheduler, *args, **kwargs):
            launches.append(True)
            return original_launch(scheduler, *args, **kwargs)

        with patch.object(GpuScheduler, "_launch", counted):
            results = batch.run(progress=False)
        self.assertTrue(all(item.status is Status.SUCCEEDED for item in results))
        self.assertLessEqual(len(launches), 1)
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
                        "seed": 0,
                        "markers": temporary,
                        "item": 0,
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
        batch = self.make_batch(count=2, delay=1)
        original = Memory.sample

        def interrupt(monitor):
            if list(self.root.glob("*.checkpoint")):
                raise KeyboardInterrupt
            return original(monitor)

        with (
            patch.object(Memory, "sample", interrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            batch.run(progress=False)
        self.assertEqual(len(list(self.root.glob("*.finally"))), 0)
        started = [
            result
            for result in batch.results
            if (self.root / f"{result.run_id}.started").exists()
        ]
        self.assertTrue(started)
        self.assertTrue(all(result.status is Status.CANCELLED for result in started))
        for result in started:
            pid = int((self.root / f"{result.run_id}.started").read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
        resumed = Batch.resume(Pipeline([GpuWork]), batch.output_dir)
        results = resumed.run(progress=False)
        self.assertTrue(all(item.status is Status.SUCCEEDED for item in results))
        saved = [item.output["saved"] for item in results]
        self.assertEqual(len(saved), 2)
        self.assertTrue(all(value in (1, 2) for value in saved))
        self.assertIn(2, saved)
        self.assertTrue(all(len(item.attempts) in (1, 2) for item in results))
        self.assertEqual(max(len(item.attempts) for item in results), 2)
        self.assertEqual(max(resumed._stage_attempts.values()), 2)

    def test_worker_results_are_read_back_instead_of_retained(self):
        batch = self.make_batch(count=1, devices=[0], delay=0.05)
        result = batch.run(progress=False)[0]
        attempt = result.attempts[-1]
        self.assertIsNone(attempt._output)
        self.assertIsNotNone(attempt.output_source)
        self.assertEqual(attempt.output["device"], "GPU-test-0")
        self.assertEqual(result.output["device"], "GPU-test-0")

    def test_low_vram_ratio_alone_does_not_shed_attempts(self):
        batch = self.make_batch(count=1, devices=[0], delay=0)
        scheduler = GpuScheduler(batch)
        memory = {0: DeviceMemory("GPU-test-0", 1000, 5)}
        host = HostMemory(64 * 1024**2, 48 * 1024**2, 0, 0)
        scheduler._relieve(memory, host)
        self.assertTrue(scheduler.host_gate.can_launch(host, []))

    def test_host_pressure_does_not_shed_the_only_running_attempt(self):
        # Shedding distributes a shared host between attempts. With one attempt
        # there is nothing to distribute, and ending it discards work no other
        # attempt is competing with while the worker's cgroup is what bounds
        # the host. A shortage with one worker pauses launches instead.
        batch = self.make_batch(count=1, devices=[0], delay=0)
        scheduler = GpuScheduler(batch)
        short = HostMemory(1000, 5, 0, 0)
        shed = []
        with patch.object(GpuScheduler, "_shed", side_effect=shed.append):
            scheduler.workers["only"] = Mock(started=0.0)
            scheduler._relieve({}, short)
            self.assertEqual(shed, [])
            self.assertFalse(scheduler.host_gate.can_launch(short, []))

            # A second attempt is over-commitment, so the newest one goes.
            scheduler.workers["newer"] = Mock(started=1.0)
            scheduler._relieve({}, short)
            self.assertEqual(shed, ["newer"])

            # An unreadable host is the same shortage, not a reason to stop the
            # one attempt that is already running.
            scheduler.workers.pop("newer")
            scheduler._relieve({}, None)
            self.assertEqual(shed, ["newer"])

    def test_cold_start_launches_one_attempt_until_every_stage_has_completed(self):
        batch = self.make_batch(count=4, devices=[0, 1], delay=0)
        scheduler = GpuScheduler(batch)
        memory = Memory([0, 1]).sample()
        host = Host().sample()
        self.assertTrue(scheduler._cold_start())
        launched = []
        with patch.object(
            GpuScheduler, "_launch", side_effect=lambda *a: launched.append(a)
        ):
            scheduler._launch_available(memory, host)
        self.assertEqual(len(launched), 1)
        self.assertEqual(launched[0][3].run_id, scheduler._warmup_run())

        # One completed record per Stage is what releases concurrent packing.
        for stage_index in range(len(batch.experiments[0].pipeline.stages)):
            batch._stage_history.append(
                {
                    "run_id": batch.experiments[0].run_id,
                    "stage": stage_index,
                    "status": Status.SUCCEEDED.value,
                }
            )
        self.assertFalse(scheduler._cold_start())
        plan = scheduler._launch_plan(memory, host)
        self.assertGreater(len(plan), 1)
        launched.clear()
        with patch.object(
            GpuScheduler, "_launch", side_effect=lambda *a: launched.append(a)
        ):
            scheduler._launch_available(memory, host)
        self.assertEqual(len(launched), len(plan))

    def test_warmup_walks_one_pipeline_to_its_later_stages(self):
        # The queue rotates as Stages finish. Following its head would measure
        # every run's first Stage before any run reached a second one, which is
        # why the warm-up run is the one furthest along instead.
        path = self.root / "warmup-order.yaml"
        path.write_text(
            f"device: [0, 1]\nseed: 0\nmarkers: {self.root}\noffset: 10\n"
            "item: !choice [1, 2, 3]\n"
        )
        batch = Batch(
            Pipeline([Prepare, Consume]), path, output_dir=self.root / "warmup-order"
        )
        scheduler = GpuScheduler(batch)
        first = batch._queue[0]
        batch._queue.remove(first)
        batch._queue.append(first)  # A finished Stage re-queues the run at the tail.
        batch._stage_progress[first] = 1

        self.assertEqual(scheduler._warmup_run(), first)
        plan = scheduler._launch_plan(
            Memory([0, 1]).sample(), Host().sample(), warmup_only=True
        )
        self.assertEqual([candidate.stage_index for candidate, _ in plan], [1])
        self.assertEqual([candidate.run_id for candidate, _ in plan], [first])

    def test_cold_start_holds_launches_while_an_attempt_is_alive(self):
        batch = self.make_batch(count=4, devices=[0, 1], delay=0)
        scheduler = GpuScheduler(batch)
        scheduler.workers["running"] = Mock(
            started=0.0, peak_kb=0.0, resident_kb=0.0, limit=None
        )
        launched = []
        with patch.object(
            GpuScheduler, "_launch", side_effect=lambda *a: launched.append(a)
        ):
            scheduler._launch_available(Memory([0, 1]).sample(), Host().sample())
        self.assertEqual(launched, [])

    def test_a_failed_first_attempt_still_measures_its_stage(self):
        # A Stage that never fits its ceiling must not run without one forever:
        # a limit-free attempt is retried outside the failure budget, so only the
        # recorded attempt can end the uncapped phase.
        batch = self.make_batch(count=1, devices=[0], delay=0)
        scheduler = GpuScheduler(batch)
        self.assertTrue(scheduler._cold_start())
        batch._stage_history.append(
            {
                "run_id": batch.experiments[0].run_id,
                "stage": 0,
                "status": Status.CANCELLED.value,
                "peak_kb": 0.0,
            }
        )
        self.assertTrue(scheduler._stage_measured(0))
        self.assertFalse(scheduler._cold_start())

    def test_cold_start_leaves_device_memory_uncapped(self):
        batch = self.make_batch(count=2, devices=[0], delay=0)
        ceilings = _captured_gpu_ceilings(batch)
        self.assertTrue(all(item.status is Status.SUCCEEDED for item in batch.results))
        # The Stage's first attempt ran without a PyTorch allocator ceiling.
        self.assertIsNone(ceilings[0])

    def test_a_measured_stage_is_capped_on_device_memory(self):
        batch = self.warm(self.make_batch(count=1, devices=[0], delay=0))
        ceilings = _captured_gpu_ceilings(batch)
        self.assertTrue(all(item.status is Status.SUCCEEDED for item in batch.results))
        self.assertEqual(len(ceilings), 1)
        self.assertGreater(float(ceilings[0]), 0)

    def test_every_stage_completing_releases_cold_start_serialisation(self):
        path = self.root / "warmup.yaml"
        path.write_text(
            f"device: [0, 1]\nseed: 0\nmarkers: {self.root}\noffset: 10\n"
            "item: !choice [1, 2, 3]\n"
        )
        batch = Batch(
            Pipeline([Prepare, Consume]), path, output_dir=self.root / "warmup"
        )
        started = []
        original = GpuScheduler._launch

        def watched(self, device, memory, host, candidate):
            # Sampled at the launch itself: a launch is the only moment that can
            # put a second attempt beside a live one.
            started.append((len(self.workers), self._cold_start()))
            return original(self, device, memory, host, candidate)

        with patch.object(GpuScheduler, "_launch", watched):
            results = batch.run(progress=False)

        self.assertTrue(all(item.status is Status.SUCCEEDED for item in results))
        # Nothing started beside a live attempt while a Stage was unmeasured.
        self.assertTrue(all(running == 0 for running, cold in started if cold))
        self.assertFalse(GpuScheduler(batch)._cold_start())

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

    def test_host_memory_pressure_sheds_newest_attempt_and_refills_on_recovery(self):
        batch = self.warm(
            self.make_batch(count=2, devices=[0, 1], delay=0, wait_for_release=True)
        )
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
        # Two workers were active, so the shed worker could otherwise have been
        # replaced immediately; host pressure must hold every refill globally.
        self.assertEqual(len(cards), 2)
        # The next healthy reading refills without waiting for the survivor.
        self.assertGreater(max(held), min(held))
        shed = [
            result
            for result in results
            if result.attempts[0].error_type == "MemoryPressure"
        ]
        self.assertTrue(shed)
        for result in shed:
            self.assertEqual(result.attempts[0].status, Status.CANCELLED)
            self.assertEqual(result.status, Status.SUCCEEDED)
            self.assertEqual(len(result.attempts), 2)
        self.assertTrue(all(item.status is Status.SUCCEEDED for item in results))

    def test_observed_peak_is_recorded_and_changes_later_admission(self):
        batch = self.make_batch(count=1, devices=[0], delay=0.05)
        batch.run(progress=False)
        peaks = [item["peak_kb"] for item in batch._gpu_history]
        self.assertTrue(peaks)
        self.assertTrue(all(value > 0 for value in peaks))
        scheduler = GpuScheduler(batch)
        self.assertGreater(scheduler.peak_ceiling, 0)

    def test_failed_host_query_sheds_newest_attempt_and_pauses_launches(self):
        batch = self.warm(self.make_batch(count=2, devices=[0], delay=0.05))
        original = Host.sample
        original_launch = GpuScheduler._launch
        launches = []
        failures = 0
        failed_tick = False

        def counted(self, device, memory, host, candidate):
            if failed_tick:
                self.fail("a failed query launched work in the same tick")
            launches.append(device)
            return original_launch(self, device, memory, host, candidate)

        def failing(monitor):
            nonlocal failed_tick, failures
            if failures < 2 and batch._active_gpu:
                failures += 1
                failed_tick = True
                raise MemoryObservationError("meminfo unavailable")
            failed_tick = False
            return original(monitor)

        with (
            patch.object(Host, "sample", failing),
            patch.object(GpuScheduler, "_launch", counted),
            self.assertWarns(RuntimeWarning),
        ):
            results = batch.run(progress=False)
        self.assertEqual(failures, 2)
        shed = [
            attempt
            for result in results
            for attempt in result.attempts
            if attempt.error_type == "MemoryPressure"
        ]
        self.assertGreaterEqual(len(shed), 1)
        self.assertLessEqual(len(shed), 2)
        self.assertTrue(all(item.status is Status.SUCCEEDED for item in results))

    def test_failed_memory_query_pauses_launches_without_stopping_the_batch(self):
        batch = self.make_batch(count=2, devices=[0, 1], delay=0.05)
        original = Memory.sample
        original_launch = GpuScheduler._launch
        launches = []
        failures = 0

        def counted(self, device, memory, host, candidate):
            launches.append(device)
            return original_launch(self, device, memory, host, candidate)

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
        self.assertTrue(launches)
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


class DeviceTests(unittest.TestCase):
    def test_device_and_seed_are_required_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ConfigError, "device is required"):
                Batch(Pipeline([]), {"seed": 0}, output_dir=root / "no-device")
            with self.assertRaisesRegex(ValueError, "seed is required"):
                Batch(Pipeline([]), {"device": [0]}, output_dir=root / "no-seed")
            with self.assertRaisesRegex(ValueError, "seed is required"):
                Experiment(Pipeline([]), {}, output_dir=root / "experiment-no-seed")
            with self.assertRaisesRegex(ConfigError, "nonempty"):
                Batch(
                    Pipeline([]),
                    {"device": [], "seed": 0},
                    output_dir=root / "empty-device",
                )
            self.assertFalse((root / "no-device").exists())
            self.assertFalse((root / "no-seed").exists())
            self.assertFalse((root / "experiment-no-seed").exists())
            self.assertFalse((root / "empty-device").exists())

    def test_device_is_a_batch_wide_plain_list(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for value in (None, 0, [True], [-1], [0, 0], ["0"]):
                with self.subTest(value=value), self.assertRaises(ConfigError):
                    Batch(
                        Pipeline([]),
                        {"device": value, "seed": 0},
                        output_dir=root / "unused",
                    )
            cfg = root / "experiment.yaml"
            cfg.write_text("device: !choice [[0], [1]]\nseed: 0\n")
            with self.assertRaises(ConfigError):
                Batch(Pipeline([]), cfg, output_dir=root / "unused")
            cfg.write_text("device: [0, 1]\nseed: 0\nx: !choice [1, 2]\n")
            batch = Batch(Pipeline([]), cfg, output_dir=root / "batch")
            self.assertEqual(len(batch.experiments), 2)
            self.assertEqual(batch.devices, (0, 1))

    def test_host_gate_uses_only_the_current_observation(self):
        gate = HostGate()
        roomy = HostMemory(64 * 1024**2, 48 * 1024**2, 0, 0)
        self.assertTrue(gate.can_launch(roomy, ["a"]))
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

    def test_swap_occupancy_does_not_gate_launches(self):
        # Swapped pages stay out until something touches them, so a host that
        # swapped once keeps reporting a partly occupied swap long after the
        # shortage. Only the memory available right now may refuse a launch.
        gate = HostGate()
        large = 64 * 1024**2
        half_swapped = HostMemory(large, large // 2, 4096, 2048)
        self.assertTrue(gate.can_launch(half_swapped, []))
        # Swap fully occupied, memory perfectly available: still a launch.
        self.assertTrue(gate.can_launch(HostMemory(large, large // 2, 4096, 0), []))
        short = HostMemory(large, HOST_RESERVE_KB - 1, 4096, 0)
        self.assertFalse(gate.can_launch(short, []))

    def test_fits_reserve_needs_headroom_above_the_reserve(self):
        available = HOST_RESERVE_KB + 2 * 1024**2
        roomy = HostMemory(64 * 1024**2, available, 0, 0)
        self.assertTrue(fits_reserve(roomy, 2 * 1024**2))
        self.assertFalse(fits_reserve(roomy, 2 * 1024**2 + 1))
        self.assertFalse(fits_reserve(HostMemory(64 * 1024**2, 0, 0, 0), 1))
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

    def test_host_memory_reports_availability_and_swap_occupancy(self):
        with tempfile.TemporaryDirectory() as temporary:
            meminfo = Path(temporary) / "meminfo"
            meminfo.write_text(
                "MemTotal:       16777216 kB\n"
                "MemAvailable:    8388608 kB\n"
                "SwapTotal:       4194304 kB\n"
                "SwapFree:        2097152 kB\n"
            )
            monitor = MeminfoMonitor(meminfo)
            observation = monitor.sample()
            self.assertEqual(observation.swap_free_kb, 2097152.0)
            self.assertEqual(observation.headroom_kb, 8388608.0 - HOST_RESERVE_KB)
            self.assertFalse(observation.tight)
            # Half the swap stays occupied, and it changes nothing: the occupied
            # pages are not memory anything can be asked to give back.
            self.assertEqual(monitor.sample().swap_free_kb, 2097152.0)
            self.assertFalse(monitor.sample().tight)

    def test_host_memory_reports_a_shortage_from_availability_alone(self):
        with tempfile.TemporaryDirectory() as temporary:
            meminfo = Path(temporary) / "meminfo"
            meminfo.write_text(
                "MemTotal:       16777216 kB\n"
                "MemAvailable:       1024 kB\n"
                "SwapTotal:       4194304 kB\n"
                "SwapFree:        2097152 kB\n"
            )
            observation = MeminfoMonitor(meminfo).sample()
            self.assertTrue(observation.tight)
            self.assertFalse(HostGate().can_launch(observation, []))

    @unittest.skipUnless(Path("/proc/meminfo").exists(), "host meminfo is unavailable")
    def test_default_host_monitor_reads_the_running_host(self):
        observation = MeminfoMonitor().sample()
        self.assertGreater(observation.total_kb, 0)
        self.assertTrue(0 <= observation.available_ratio <= 1)


if __name__ == "__main__":
    unittest.main()
