import os
import sqlite3
import subprocess
from itertools import pairwise
from pathlib import Path
from time import monotonic, sleep
from unittest.mock import Mock, patch

import pytest
import yaml

from expman import Batch, ConfigError, Pipeline, Stage, Status, devices, environment
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
from expman.experiment import Experiment
from expman.limits import cap_kb
from expman.scheduling import GpuScheduler, HostGate
from expman.storage import PickleSerializer, read_record

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

    def init(self, ctx):
        self.saved = 0

    def loop(self, data, ctx, index, max_iter=2):
        root = Path(ctx.cfg["markers"])
        if index == 0:
            (root / f"{ctx.run_id}.started").write_text(str(os.getpid()))
            self.saved += 1
            (root / f"{ctx.run_id}.checkpoint").touch()
            return None
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
                "saved": self.saved,
                "cfg_devices": list(ctx.cfg["device"]),
            }
        finally:
            (root / f"{ctx.run_id}.finally").touch()


class SharedWork(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {}

    def init(self, ctx):
        self.producer = None

    def process(self, data, ctx):
        self.producer = os.getpid()
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


def _captured_worker_env(batch):
    """Run the Batch, returning the environment each worker was started with."""
    envs = []
    original = subprocess.Popen

    def captured(*args, **kwargs):
        argv = args[0] if args else kwargs.get("args", ())
        if "expman._worker" in argv:
            envs.append(kwargs.get("env", {}))
        return original(*args, **kwargs)

    with patch("expman.scheduling.subprocess.Popen", side_effect=captured):
        batch.run(progress=False)
    return envs


def make_batch(root, *, count=3, devices=None, **options):
    cfg = {
        "device": [0, 1] if devices is None else devices,
        "seed": 0,
        "markers": str(root),
        "crash": False,
        "fail_once": False,
        "wait_for_release": False,
        "delay": 0.1,
        "item": 0,
        **options,
    }
    path = root / "experiments.yaml"
    path.write_text(yaml.safe_dump(cfg) + f"grid:\n  item: {list(range(count))}\n")
    return Batch(Pipeline([GpuWork]), path, output_dir=root / "batch")


def warm(batch, *, gpu_peak_kb=1024, peak_kb=4096):
    """Give every Stage the completed sample a previous run would have left.

    An unmeasured Stage probes at a bounded share of the device, and a Stage
    that has never reported CUDA usage receives no GPU contract afterwards.
    Tests about packing therefore start from a Batch whose Stages have
    already been measured, with a CUDA peak on record.
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


@pytest.mark.skipif(os.name != "posix", reason="POSIX worker process groups")
class TestGpuScheduling:
    @pytest.fixture(autouse=True)
    def fake_devices(self, monkeypatch):
        monkeypatch.setattr(devices, "NvidiaMemory", Memory)
        monkeypatch.setattr(devices, "MeminfoMonitor", Host)
        monkeypatch.setattr(devices, "POLL_INTERVAL", 0.01)

    def test_worker_environment_gets_the_framework_defaults(
        self, tmp_path, monkeypatch
    ):
        """The thread ceiling and the HF mirror reach every worker untouched."""
        for name in (*environment.THREAD_VARIABLES, "HF_ENDPOINT"):
            monkeypatch.delenv(name, raising=False)
        batch = warm(make_batch(tmp_path, count=1))

        envs = _captured_worker_env(batch)

        assert envs, "no worker process was launched"
        assert envs[0]["OMP_NUM_THREADS"] == str(environment.WORKER_THREADS)
        assert envs[0]["MKL_NUM_THREADS"] == str(environment.WORKER_THREADS)
        assert envs[0]["HF_ENDPOINT"] == environment.HF_MIRROR
        # CUDA_VISIBLE_DEVICES stays the scheduler's own decision.
        assert "expman.environment" not in envs[0]
        recorded = read_record(
            next(batch.output_dir.rglob("bootstrap.pkl")), PickleSerializer()
        )["environment"]
        assert recorded["HF_ENDPOINT"] == environment.HF_MIRROR
        assert recorded["OMP_NUM_THREADS"] == str(environment.WORKER_THREADS)

    def test_multiple_devices_and_metrics_results_are_collected(self, tmp_path):
        batch = warm(make_batch(tmp_path, count=4, delay=0.25))
        results = batch.run(progress=False)
        assert all(item.status is Status.SUCCEEDED for item in results)
        assert {item.output["device"] for item in results} == {
            "GPU-test-0",
            "GPU-test-1",
        }
        assert len({item.output["pid"] for item in results}) == 4
        assert all(
            item.output["import_device"] == item.output["device"] for item in results
        )
        assert all(item.output["cfg_devices"] == [0, 1] for item in results)
        assert any(item["concurrency"] > 1 for item in batch._gpu_history)
        measured = [item for item in batch._stage_history if not item.get("reused")]
        assert measured
        assert all(item["peak_kb"] > 0 for item in measured)
        assert batch._host_memory["available_ratio"] == 0.75
        assert batch._host_memory["swap_free_kb"] == 0.0
        assert not batch._host_memory["tight"]
        with sqlite3.connect(batch.output_dir / "metrics.sqlite3") as db:
            assert db.execute("SELECT count(*) FROM metrics").fetchone()[0] == 40
        assert batch.recorder.events
        resumed = Batch.resume(Pipeline([GpuWork]), batch.output_dir)
        assert [item.output for item in resumed.run(progress=False)] == [
            item.output for item in results
        ]
        assert resumed.devices == (0, 1)

    def test_top_level_stages_run_independently_and_record_dependencies(self, tmp_path):
        path = tmp_path / "two-stages.yaml"
        path.write_text(
            f"device: [0]\nseed: 0\nmarkers: {tmp_path}\noffset: 10\nitem: 1\n"
            "grid:\n  item: [1, 2]\n"
        )
        batch = Batch(
            Pipeline([Prepare, Consume]), path, output_dir=tmp_path / "two-stages"
        )

        results = batch.run(progress=False)

        assert [result.output for result in results] == [11, 12]
        assert sorted(entry["stage"] for entry in batch._stage_history) == [0, 0, 1, 1]
        assert all(
            (tmp_path / f"{result.run_id}.prepare").exists() for result in results
        )
        assert all(
            (tmp_path / f"{result.run_id}.consume").exists() for result in results
        )
        dependencies = {
            entry["stage"]: {path for path, _digest in entry["dependencies"]}
            for entry in batch._stage_history
        }
        assert dependencies[0] == {("item",), ("markers",)}
        assert dependencies[1] == {("markers",), ("offset",)}
        scheduler = GpuScheduler(batch)
        assert {
            path
            for item in batch.experiments
            for path in scheduler._prefix_paths(item.run_id, 1)
        } == {("item",), ("markers",), ("offset",)}
        predicted = scheduler._stage_duration(results[0].run_id, 1)
        assert predicted is not None
        assert 0 < predicted.lower <= predicted.center <= predicted.upper

    def test_completed_zero_gpu_sample_needs_no_vram_reservation(self, tmp_path):
        batch = make_batch(tmp_path, count=2, devices=[0])
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

        assert (
            scheduler._stage_peak(
                batch.experiments[1].run_id, 0, "gpu_peak_kb", GPU_PEAK_KB_DEFAULT
            )
            == 0.0
        )

    def test_gpu_reservations_and_ceilings_never_exceed_card_capacity(self, tmp_path):
        batch = make_batch(tmp_path, count=2, devices=[0])
        scheduler = GpuScheduler(batch)
        capacity_kb = 3 * 1024

        with patch.object(scheduler, "_stage_peak", return_value=100 * capacity_kb):
            candidates = scheduler._stage_candidates(0, capacity_kb)

        assert candidates
        assert all(item.gpu_reserve_kb == capacity_kb for item in candidates)
        assert all(item.gpu_cap_kb == capacity_kb for item in candidates)

    def test_admission_charges_the_estimate_while_the_cgroup_adds_the_margin(
        self, tmp_path
    ):
        # The allowance is what a single attempt may reach before it is retried,
        # not capacity a packed plan may spend: charging it at admission too would
        # idle memory that another attempt could use.
        batch = make_batch(tmp_path, count=2, devices=[0])
        scheduler = GpuScheduler(batch)
        estimate = 4 * 1024**2

        with patch.object(scheduler, "_stage_peak", return_value=estimate):
            candidates = scheduler._stage_candidates(0, 8 * 1024**2)

        assert candidates
        assert all(item.host_reserve_kb == estimate for item in candidates)
        assert all(
            cap_kb(item.host_reserve_kb) > item.host_reserve_kb for item in candidates
        )

    def test_gpu_admission_charges_the_reserve_under_the_allocator_ceiling(
        self, tmp_path
    ):
        batch = make_batch(tmp_path, count=2, devices=[0])
        scheduler = GpuScheduler(batch)
        capacity_kb = 1024

        with patch.object(scheduler, "_stage_peak", return_value=capacity_kb // 2):
            candidates = scheduler._stage_candidates(0, capacity_kb)

        assert candidates
        assert all(item.gpu_reserve_kb == capacity_kb // 2 for item in candidates)
        assert all(
            item.gpu_cap_kb == capacity_kb // 2 * devices.CAPACITY_FACTOR
            for item in candidates
        )

    def test_a_running_worker_is_charged_up_to_its_reservation(self, tmp_path):
        batch = make_batch(tmp_path, count=1, devices=[0], delay=0)
        scheduler = GpuScheduler(batch)
        scheduler.workers["running"] = Mock(
            started=0.0, peak_kb=0.0, resident_kb=0.0, host_reserve_kb=2 * 1024**2
        )
        host = HostMemory(64 * 1024**2, HOST_RESERVE_KB + 3 * 1024**2, 0, 0)

        assert scheduler._admits(host, 1024**2)
        assert not scheduler._admits(host, 2 * 1024**2)

    def test_known_stage_group_has_only_one_runnable_representative(self, tmp_path):
        path = tmp_path / "groups.yaml"
        path.write_text(
            f"device: [0]\nseed: 0\nmarkers: {tmp_path}\nunused: 1\n"
            "grid:\n  unused: [1, 2, 3]\n"
        )
        batch = Batch(Pipeline([SharedWork]), path, output_dir=tmp_path / "groups")
        scheduler = GpuScheduler(batch)

        candidates = scheduler._stage_candidates(0, 16 * 1024**2)

        assert len(candidates) == 1

    def test_stage_group_follows_declared_dependencies(self, tmp_path):
        path = tmp_path / "declared.yaml"
        path.write_text(
            f"device: [0]\nseed: 0\nmarkers: {tmp_path}\nitem: 1\n"
            "grid:\n  item: [1, 2]\n"
            "crash: false\nfail_once: false\nwait_for_release: false\ndelay: 0\n"
        )
        batch = Batch(Pipeline([GpuWork]), path, output_dir=tmp_path / "declared")
        scheduler = GpuScheduler(batch)
        first, second = (item.run_id for item in batch.experiments)

        # GpuWork declares item, so its two values are separate groups.
        assert scheduler._stage_group(first, 0) != scheduler._stage_group(second, 0)
        assert {
            path
            for item in batch.experiments
            for path in scheduler._prefix_paths(item.run_id, 0)
        } == {
            ("markers",),
            ("item",),
            ("crash",),
            ("fail_once",),
            ("wait_for_release",),
            ("delay",),
            ("device",),
        }

    def test_undeclared_configuration_does_not_split_stage_groups(self, tmp_path):
        path = tmp_path / "undeclared.yaml"
        path.write_text(
            f"device: [0]\nseed: 0\nmarkers: {tmp_path}\nitem: 0\n"
            "crash: false\nfail_once: false\nwait_for_release: false\ndelay: 0\n"
            "unused: 1\ngrid:\n  unused: [1, 2]\n"
        )
        batch = Batch(Pipeline([GpuWork]), path, output_dir=tmp_path / "undeclared")
        scheduler = GpuScheduler(batch)
        first, second = (item.run_id for item in batch.experiments)

        # unused is not declared, so the two runs share one reusable group.
        assert scheduler._stage_group(first, 0) == scheduler._stage_group(second, 0)

    def test_conditional_dependencies_follow_the_config_value(self, tmp_path):
        path = tmp_path / "conditional.yaml"
        path.write_text(
            "device: [0]\nseed: 0\n"
            "Baseline: B1\nimputer: i\npredictor: p1\n"
            "grid:\n"
            "  Baseline: [B1, B2]\n  predictor: [p1, p2]\n"
        )
        batch = Batch(
            Pipeline([ConditionalWork]), path, output_dir=tmp_path / "conditional"
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
        assert groups[("B1", "p1")] == groups[("B1", "p2")]
        assert groups[("B2", "p1")] != groups[("B2", "p2")]
        assert groups[("B1", "p1")] != groups[("B2", "p1")]

    def test_tick_greedily_fills_a_card(self, tmp_path):
        batch = warm(make_batch(tmp_path, count=4, devices=[0]))
        scheduler = GpuScheduler(batch)
        started = []

        def launch(device, memory, host, candidate):
            started.append(device)
            batch._queue.remove(candidate.run_id)
            return True

        with patch.object(scheduler, "_launch", side_effect=launch):
            scheduler._launch_available(Memory([0]).sample(), Host().sample())

        assert started == [0, 0, 0, 0]

    def test_tick_plan_combines_memory_and_gpu_stages(self, tmp_path):
        path = tmp_path / "two-stage.yaml"
        path.write_text(
            f"device: [0]\nseed: 0\nmarkers: {tmp_path}\n"
            "crash: false\nfail_once: false\nwait_for_release: false\ndelay: 0\n"
            "item: 0\ngrid:\n  item: [0, 1, 2, 3, 4, 5, 6, 7]\n"
        )
        batch = Batch(Pipeline([GpuWork, GpuWork]), path, output_dir=tmp_path / "plan")
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
        # One joint plan, not a barrier per Stage: both Stages share this tick.
        assert set(stages) == {0, 1}
        # The card holds 3000 MiB, so a third 1 MiB Stage 1 reservation does not
        # fit and every remaining slot goes to the Stage that reserves none.
        assert stages.count(0) == 4
        assert stages.count(1) == 2

    def test_shared_prefix_is_materialized_without_worker_per_member(self, tmp_path):
        # SharedWork declares no dependencies, so every Batch member shares one
        # reusable group and receives a local cache reference without a worker.
        output_dir = tmp_path / "shared-two"
        path = tmp_path / "shared.yaml"
        path.write_text("device: [0]\nseed: 0\nunused: 1\ngrid:\n  unused: [1, 2]\n")
        batch = Batch(Pipeline([SharedWork]), path, output_dir=output_dir)
        seeded = Experiment(
            Pipeline([SharedWork]),
            {"seed": 0, "unused": 0},
            output_dir=tmp_path / "seed",
            _cache_root=output_dir / "cache",
        )
        seeded.run_stage(0, attempt=1)
        expected = seeded._store.completed((0,))["output"]
        launches = []
        original_launch = GpuScheduler._launch

        def counted(scheduler, *args, **kwargs):
            launches.append(True)
            return original_launch(scheduler, *args, **kwargs)

        with patch.object(GpuScheduler, "_launch", counted):
            results = batch.run(progress=False)
        assert all(item.status is Status.SUCCEEDED for item in results)
        assert len(launches) <= 1
        assert [item.output for item in results] == [expected, expected]
        references = [
            experiment._store.completed_reference((0,))
            for experiment in batch.experiments
        ]
        assert references[0] == references[1]
        assert (
            batch.experiments[1]._store.completed((0,))["state"]["producer"] == expected
        )

    @pytest.mark.parametrize("crash", [False, True], ids=["fail-once", "crash"])
    def test_failure_and_unexpected_process_exit_get_one_retry(self, tmp_path, crash):
        batch = Batch(
            Pipeline([GpuWork]),
            {
                "device": [0],
                "seed": 0,
                "markers": str(tmp_path),
                "item": 0,
                "wait_for_release": False,
                "delay": 0.1,
                "crash": crash,
                "fail_once": not crash,
            },
            output_dir=tmp_path / "batch",
        )
        result = batch.run(progress=False)[0]
        assert len(result.attempts) == 2
        assert result.status is (Status.FAILED if crash else Status.SUCCEEDED)
        assert result.attempts[0].error_type == (
            "WorkerExit" if crash else "ValueError"
        )

    def test_interrupt_kills_all_without_finally_and_restores_checkpoint(
        self, tmp_path
    ):
        batch = make_batch(tmp_path, count=2, delay=1)
        original = Memory.sample

        def interrupt(monitor):
            if list(tmp_path.glob("*.checkpoint")):
                raise KeyboardInterrupt
            return original(monitor)

        with (
            patch.object(Memory, "sample", interrupt),
            pytest.raises(KeyboardInterrupt),
        ):
            batch.run(progress=False)
        assert len(list(tmp_path.glob("*.finally"))) == 0
        started = [
            result
            for result in batch.results
            if (tmp_path / f"{result.run_id}.started").exists()
        ]
        assert started
        assert all(result.status is Status.CANCELLED for result in started)
        for result in started:
            pid = int((tmp_path / f"{result.run_id}.started").read_text())
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
        resumed = Batch.resume(Pipeline([GpuWork]), batch.output_dir)
        results = resumed.run(progress=False)
        assert all(item.status is Status.SUCCEEDED for item in results)
        # The state attribute comes from the checkpoint, not a re-executed loop.
        saved = [item.output["saved"] for item in results]
        assert len(saved) == 2
        assert all(value == 1 for value in saved)
        assert all(len(item.attempts) in (1, 2) for item in results)
        assert max(len(item.attempts) for item in results) == 2
        assert max(resumed._stage_attempts.values()) == 2

    def test_worker_results_are_read_back_instead_of_retained(self, tmp_path):
        batch = make_batch(tmp_path, count=1, devices=[0], delay=0.05)
        result = batch.run(progress=False)[0]
        attempt = result.attempts[-1]
        assert attempt.output_source is not None
        assert attempt.output["device"] == "GPU-test-0"
        assert result.output["device"] == "GPU-test-0"

    def test_low_vram_ratio_alone_does_not_shed_attempts(self, tmp_path):
        batch = make_batch(tmp_path, count=1, devices=[0], delay=0)
        scheduler = GpuScheduler(batch)
        host = HostMemory(64 * 1024**2, 48 * 1024**2, 0, 0)
        scheduler._relieve(host)
        assert scheduler.host_gate.can_launch(host)

    def test_host_pressure_does_not_shed_the_only_running_attempt(self, tmp_path):
        # Shedding distributes a shared host between attempts. With one attempt
        # there is nothing to distribute, and ending it discards work no other
        # attempt is competing with while the worker's cgroup is what bounds
        # the host. A shortage with one worker pauses launches instead.
        batch = make_batch(tmp_path, count=1, devices=[0], delay=0)
        scheduler = GpuScheduler(batch)
        short = HostMemory(1000, 5, 0, 0)
        shed = []
        with patch.object(GpuScheduler, "_shed", side_effect=shed.append):
            scheduler.workers["only"] = Mock(started=0.0)
            scheduler._relieve(short)
            assert shed == []
            assert not scheduler.host_gate.can_launch(short)

            # A second attempt is over-commitment, so the newest one goes.
            scheduler.workers["newer"] = Mock(started=1.0)
            scheduler._relieve(short)
            assert shed == ["newer"]

            # An unreadable host is the same shortage, not a reason to stop the
            # one attempt that is already running.
            scheduler.workers.pop("newer")
            scheduler._relieve(None)
            assert shed == ["newer"]

    def test_an_unmeasured_batch_probes_within_a_host_share(self, tmp_path):
        batch = make_batch(tmp_path, count=4, devices=[0, 1], delay=0)
        scheduler = GpuScheduler(batch)
        memory = Memory([0, 1]).sample()
        host = Host().sample()
        probe_kb = devices.probe_cap_kb(
            host.total_kb, devices.HOST_PEAK_KB_DEFAULT, scheduler.peaks.ceiling_kb
        )
        launched = []
        with patch.object(
            GpuScheduler,
            "_launch",
            side_effect=lambda *a: launched.append(a[3]),
        ):
            scheduler._launch_available(memory, host)
        # Each probe is priced at a quarter of the host, so the packing
        # arithmetic itself bounds concurrency; nothing serialises the probes.
        assert len(launched) == 2
        assert {candidate.host_reserve_kb for candidate in launched} == {probe_kb}

        # Measured peaks replace the probe pricing, and every run fits at once.
        warm(batch)
        fresh = GpuScheduler(batch)
        launched.clear()
        with patch.object(
            GpuScheduler, "_launch", side_effect=lambda *a: launched.append(a)
        ):
            fresh._launch_available(memory, host)
        assert len(launched) == 4

    def test_a_live_attempt_does_not_hold_other_launches_back(self, tmp_path):
        batch = make_batch(tmp_path, count=4, devices=[0, 1], delay=0)
        scheduler = GpuScheduler(batch)
        scheduler.workers["running"] = Mock(
            started=0.0, peak_kb=0.0, resident_kb=0.0, host_reserve_kb=0.0, limit=None
        )
        launched = []
        with patch.object(
            GpuScheduler, "_launch", side_effect=lambda *a: launched.append(a)
        ):
            scheduler._launch_available(Memory([0, 1]).sample(), Host().sample())
        # The probes a live attempt used to serialise now launch beside it;
        # only the host share bounds how many.
        assert len(launched) == 2

    def test_a_probe_is_capped_on_device_memory(self, tmp_path):
        batch = make_batch(tmp_path, count=2, devices=[0], delay=0)
        ceilings = _captured_gpu_ceilings(batch)
        assert all(item.status is Status.SUCCEEDED for item in batch.results)
        # The probe runs at a quarter of the card instead of without a ceiling.
        card_kb = Memory([0]).sample()[0].total * 1024
        expected = min(
            cap_kb(devices.probe_cap_kb(card_kb, devices.GPU_PEAK_KB_DEFAULT)),
            card_kb,
        )
        assert len(ceilings) == 2
        assert [float(item) for item in ceilings] == [pytest.approx(expected)] * 2

    def test_a_measured_stage_is_capped_on_device_memory(self, tmp_path):
        batch = warm(make_batch(tmp_path, count=1, devices=[0], delay=0))
        ceilings = _captured_gpu_ceilings(batch)
        assert all(item.status is Status.SUCCEEDED for item in batch.results)
        assert len(ceilings) == 1
        assert float(ceilings[0]) > 0

    def test_probes_start_beside_live_attempts(self, tmp_path):
        path = tmp_path / "probes.yaml"
        path.write_text(
            f"device: [0, 1]\nseed: 0\nmarkers: {tmp_path}\noffset: 10\n"
            "item: 1\ngrid:\n  item: [1, 2, 3]\n"
        )
        batch = Batch(
            Pipeline([Prepare, Consume]), path, output_dir=tmp_path / "probes"
        )
        started = []
        original_launch = GpuScheduler._launch

        def watched(scheduler, device, memory, host, candidate):
            # Sampled at the launch itself: a launch is the only moment that can
            # put a second attempt beside a live one.
            started.append(len(scheduler.workers))
            return original_launch(scheduler, device, memory, host, candidate)

        with patch.object(GpuScheduler, "_launch", watched):
            results = batch.run(progress=False)

        assert all(item.status is Status.SUCCEEDED for item in results)
        # Probes go out while other attempts are still running: unmeasured
        # Stages are bounded by their share, not by serialisation.
        assert any(running > 0 for running in started)

    def test_host_memory_shortage_pauses_launches_until_it_recovers(self, tmp_path):
        batch = make_batch(tmp_path, count=1, devices=[0], delay=0.05)
        original = Host.sample
        pressured = []

        def shortage(monitor):
            if len(pressured) < 5:
                pressured.append(len(batch._active_gpu))
                return HostMemory(1000, 5, 0, 0)
            return original(monitor)

        with patch.object(Host, "sample", shortage):
            results = batch.run(progress=False)
        assert pressured == [0] * 5
        assert results[0].status is Status.SUCCEEDED

    def test_host_memory_pressure_sheds_newest_attempt_and_refills_on_recovery(
        self, tmp_path
    ):
        batch = warm(
            make_batch(
                tmp_path, count=2, devices=[0, 1], delay=0, wait_for_release=True
            )
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
                and all((tmp_path / f"{key}.checkpoint").exists() for key in active)
            ):
                dropped.append(active[-1])
                cards.extend(batch._active_gpu.values())
                return HostMemory(1000, 5, 0, 0)
            if dropped and not (tmp_path / "release").exists():
                held.append(len(batch._active_gpu))
                if len(held) >= 5:
                    (tmp_path / "release").touch()
            return original(monitor)

        with (
            patch.object(devices, "HOST_RESERVE_KB", 2 * 1024**2),
            patch.object(Host, "sample", pressure),
        ):
            results = batch.run(progress=False)
        assert dropped
        # Two workers were active, so the shed worker could otherwise have been
        # replaced immediately; host pressure must hold every refill globally.
        assert len(cards) == 2
        # The next healthy reading refills without waiting for the survivor.
        assert max(held) > min(held)
        shed = [
            result
            for result in results
            if result.attempts[0].error_type == "MemoryPressure"
        ]
        assert shed
        for result in shed:
            assert result.attempts[0].status is Status.CANCELLED
            assert result.status is Status.SUCCEEDED
            assert len(result.attempts) == 2
        assert all(item.status is Status.SUCCEEDED for item in results)

    def test_observed_peak_is_recorded_and_changes_later_admission(self, tmp_path):
        batch = make_batch(tmp_path, count=1, devices=[0], delay=0.05)
        batch.run(progress=False)
        peaks = [item["peak_kb"] for item in batch._gpu_history]
        assert peaks
        assert all(value > 0 for value in peaks)
        scheduler = GpuScheduler(batch)
        assert scheduler.peaks.ceiling_kb > 0

    def test_failed_host_query_sheds_newest_attempt_and_pauses_launches(self, tmp_path):
        batch = warm(make_batch(tmp_path, count=2, devices=[0], delay=0.05))
        original = Host.sample
        original_launch = GpuScheduler._launch
        launches = []
        failures = 0
        failed_tick = False

        def counted(scheduler, device, memory, host, candidate):
            if failed_tick:
                pytest.fail("a failed query launched work in the same tick")
            launches.append(device)
            return original_launch(scheduler, device, memory, host, candidate)

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
            pytest.warns(RuntimeWarning),
        ):
            results = batch.run(progress=False)
        assert failures == 2
        shed = [
            attempt
            for result in results
            for attempt in result.attempts
            if attempt.error_type == "MemoryPressure"
        ]
        assert 1 <= len(shed) <= 2
        assert all(item.status is Status.SUCCEEDED for item in results)

    def test_failed_memory_query_pauses_launches_without_stopping_the_batch(
        self, tmp_path
    ):
        batch = make_batch(tmp_path, count=2, devices=[0, 1], delay=0.05)
        original = Memory.sample
        original_launch = GpuScheduler._launch
        launches = []
        failures = 0

        def counted(scheduler, device, memory, host, candidate):
            launches.append(device)
            return original_launch(scheduler, device, memory, host, candidate)

        def failing(monitor):
            nonlocal failures
            if failures < 3:
                failures += 1
                assert launches == []
                raise MemoryObservationError("nvidia-smi unavailable")
            return original(monitor)

        with (
            patch.object(Memory, "sample", failing),
            patch.object(GpuScheduler, "_launch", counted),
            pytest.warns(RuntimeWarning),
        ):
            results = batch.run(progress=False)
        assert failures == 3
        assert launches
        assert all(item.status is Status.SUCCEEDED for item in results)
        assert all(len(item.attempts) == 1 for item in results)
        assert not batch._active_gpu

    def test_memory_is_rechecked_on_every_tick(self, tmp_path):
        batch = make_batch(tmp_path, count=1, devices=[0], delay=0.3)
        original = Memory.sample
        stamps = []

        def recorded(monitor):
            stamps.append(monotonic())
            return original(monitor)

        with patch.object(Memory, "sample", recorded):
            batch.run(progress=False)
        gaps = [later - earlier for earlier, later in pairwise(stamps)]
        assert len(stamps) >= 10
        assert max(gaps) < 0.1


class TestDevice:
    def test_device_and_seed_are_required_before_creating_output(self, tmp_path):
        root = tmp_path
        with pytest.raises(ConfigError, match="device is required"):
            Batch(Pipeline([]), {"seed": 0}, output_dir=root / "no-device")
        with pytest.raises(ValueError, match="seed is required"):
            Batch(Pipeline([]), {"device": [0]}, output_dir=root / "no-seed")
        with pytest.raises(ValueError, match="seed is required"):
            Experiment(Pipeline([]), {}, output_dir=root / "experiment-no-seed")
        with pytest.raises(ConfigError, match="nonempty"):
            Batch(
                Pipeline([]),
                {"device": [], "seed": 0},
                output_dir=root / "empty-device",
            )
        assert not (root / "no-device").exists()
        assert not (root / "no-seed").exists()
        assert not (root / "experiment-no-seed").exists()
        assert not (root / "empty-device").exists()

    @pytest.mark.parametrize("value", [None, 0, [True], [-1], [0, 0], ["0"]])
    def test_invalid_device_values_are_rejected(self, tmp_path, value):
        with pytest.raises(ConfigError):
            Batch(
                Pipeline([]),
                {"device": value, "seed": 0},
                output_dir=tmp_path / "unused",
            )

    def test_device_cannot_be_an_expansion_axis(self, tmp_path):
        cfg = tmp_path / "experiment.yaml"
        cfg.write_text("device: [0]\nseed: 0\ngrid:\n  device: [[0], [1]]\n")
        with pytest.raises(ConfigError):
            Batch(Pipeline([]), cfg, output_dir=tmp_path / "unused")

    def test_device_is_a_batch_wide_plain_list(self, tmp_path):
        cfg = tmp_path / "experiment.yaml"
        cfg.write_text("device: [0, 1]\nseed: 0\nx: 1\ngrid:\n  x: [1, 2]\n")
        batch = Batch(Pipeline([]), cfg, output_dir=tmp_path / "batch")
        assert len(batch.experiments) == 2
        assert batch.devices == (0, 1)

    def test_host_gate_uses_only_the_current_observation(self):
        gate = HostGate()
        roomy = HostMemory(64 * 1024**2, 48 * 1024**2, 0, 0)
        assert gate.can_launch(roomy)
        assert not gate.can_launch(None)

    def test_host_gate_refuses_room_below_the_byte_reserve(self):
        # The default reserve is 1 GiB, so the host gate refuses a reading below
        # it and admits a positive amount above it.
        gate = HostGate()
        nearly_full = HostMemory(HOST_RESERVE_KB, HOST_RESERVE_KB - 1, 0, 0)
        assert nearly_full.available_ratio > 0.9
        assert not gate.can_launch(nearly_full)
        roomy = HostMemory(64 * 1024**2, HOST_RESERVE_KB + 1, 0, 0)
        assert gate.can_launch(roomy)

    def test_swap_occupancy_does_not_gate_launches(self):
        # Swapped pages stay out until something touches them, so a host that
        # swapped once keeps reporting a partly occupied swap long after the
        # shortage. Only the memory available right now may refuse a launch.
        gate = HostGate()
        large = 64 * 1024**2
        half_swapped = HostMemory(large, large // 2, 4096, 2048)
        assert gate.can_launch(half_swapped)
        # Swap fully occupied, memory perfectly available: still a launch.
        assert gate.can_launch(HostMemory(large, large // 2, 4096, 0))
        short = HostMemory(large, HOST_RESERVE_KB - 1, 4096, 0)
        assert not gate.can_launch(short)

    def test_fits_reserve_needs_headroom_above_the_reserve(self):
        available = HOST_RESERVE_KB + 2 * 1024**2
        roomy = HostMemory(64 * 1024**2, available, 0, 0)
        assert fits_reserve(roomy, 2 * 1024**2)
        assert not fits_reserve(roomy, 2 * 1024**2 + 1)
        assert not fits_reserve(HostMemory(64 * 1024**2, 0, 0, 0), 1)
        assert not fits_reserve(None, 1)

    def test_query_timeout_is_an_observation_error_with_the_default_timeout(self):
        with (
            patch(
                "expman.devices.subprocess.run",
                side_effect=subprocess.TimeoutExpired("nvidia-smi", 3),
            ) as query,
            pytest.raises(MemoryObservationError),
        ):
            NvidiaMemory((0,)).sample()
        assert query.call_args.kwargs["timeout"] == 3.0

    def test_nvidia_smi_is_found_outside_path(self, tmp_path):
        fallback = tmp_path / "nvidia-smi"
        fallback.write_text("#!/bin/sh\n")
        with (
            patch("expman.devices.shutil.which", return_value=None),
            patch("expman.devices.NVIDIA_SMI_FALLBACKS", (fallback,)),
            patch("expman.devices.subprocess.run") as query,
        ):
            query.return_value.stdout = "0, GPU-first, 100, 10\n"
            NvidiaMemory((0,)).sample()
        assert query.call_args.args[0][0] == str(fallback)
        with patch("expman.devices.shutil.which", return_value="/usr/bin/nvidia-smi"):
            assert nvidia_smi() == "/usr/bin/nvidia-smi"

    def test_nvidia_memory_validates_observations_and_identities(self):
        with patch("expman.devices.subprocess.run") as query:
            query.return_value.stdout = (
                "0, GPU-first, 100, 10\n1, GPU-second, 200, 150\n"
            )
            monitor = NvidiaMemory((0, 1))
            assert monitor.sample()[0].free_ratio == 0.1
            assert "--id=0,1" in query.call_args.args[0]
            query.return_value.stdout = (
                "0, GPU-replaced, 100, 10\n1, GPU-second, 200, 150\n"
            )
            with pytest.raises(RuntimeError, match="identities changed"):
                monitor.sample()
            query.return_value.stdout = "0, GPU-first, N/A, N/A\n"
            with pytest.raises(RuntimeError):
                NvidiaMemory((0,)).sample()

    def test_host_memory_reads_meminfo_and_rejects_invalid_values(self, tmp_path):
        path = tmp_path / "meminfo"
        path.write_text(
            "MemTotal:       1000 kB\n"
            "MemFree:         100 kB\n"
            "MemAvailable:    250 kB\n"
            "SwapTotal:       400 kB\n"
            "SwapFree:        300 kB\n"
        )
        observation = MeminfoMonitor(path).sample()
        assert observation.available_ratio == 0.25
        assert observation.swap_total_kb == 400.0
        assert observation.swap_free_kb == 300.0
        path.write_text("MemTotal: 1000 kB\nMemAvailable: 2000 kB\n")
        with pytest.raises(MemoryObservationError, match="invalid host memory"):
            MeminfoMonitor(path).sample()
        path.write_text("MemTotal: 1000 kB\nMemAvailable: N/A\n")
        with pytest.raises(
            MemoryObservationError, match="could not observe host memory"
        ):
            MeminfoMonitor(path).sample()
        path.write_text("MemFree: 100 kB\n")
        with pytest.raises(
            MemoryObservationError, match="could not observe host memory"
        ):
            MeminfoMonitor(path).sample()

    def test_host_memory_reports_availability_and_swap_occupancy(self, tmp_path):
        meminfo = tmp_path / "meminfo"
        meminfo.write_text(
            "MemTotal:       16777216 kB\n"
            "MemAvailable:    8388608 kB\n"
            "SwapTotal:       4194304 kB\n"
            "SwapFree:        2097152 kB\n"
        )
        monitor = MeminfoMonitor(meminfo)
        observation = monitor.sample()
        assert observation.swap_free_kb == 2097152.0
        assert observation.headroom_kb == 8388608.0 - HOST_RESERVE_KB
        assert not observation.tight
        # Half the swap stays occupied, and it changes nothing: the occupied
        # pages are not memory anything can be asked to give back.
        assert monitor.sample().swap_free_kb == 2097152.0
        assert not monitor.sample().tight

    def test_host_memory_reports_a_shortage_from_availability_alone(self, tmp_path):
        meminfo = tmp_path / "meminfo"
        meminfo.write_text(
            "MemTotal:       16777216 kB\n"
            "MemAvailable:       1024 kB\n"
            "SwapTotal:       4194304 kB\n"
            "SwapFree:        2097152 kB\n"
        )
        observation = MeminfoMonitor(meminfo).sample()
        assert observation.tight
        assert not HostGate().can_launch(observation)

    @pytest.mark.skipif(
        not Path("/proc/meminfo").exists(), reason="host meminfo is unavailable"
    )
    def test_default_host_monitor_reads_the_running_host(self):
        observation = MeminfoMonitor().sample()
        assert observation.total_kb > 0
        assert 0 <= observation.available_ratio <= 1
