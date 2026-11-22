import json
import os
import sqlite3
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from time import monotonic, perf_counter, sleep
from unittest.mock import Mock, patch

import batch_helpers
import pytest
import yaml

from expman import ConfigError, Pipeline, Stage, Status, devices, environment
from expman.core.context import RunContext
from expman.core.experiment import Experiment, StageResult
from expman.resources.devices import (
    DeviceMemory,
    HostMemory,
    MeminfoMonitor,
    MemoryObservationError,
    NvidiaMemory,
    fits_reserve,
    nvidia_smi,
)
from expman.resources.limits import MemoryLimit, cap_kb
from expman.scheduling._planning import Candidate
from expman.scheduling._process import Worker
from expman.scheduling._resource import _admits
from expman.scheduling.scheduler import GpuScheduler
from expman.settings import settings
from expman.storage.base import PickleSerializer, read_record
from expman.storage.prefix_cache import PrefixCache

IMPORT_DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES")


class GpuWork(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {
            "markers": True,
            "item": True,
            "crash": True,
            "wait_for_release": True,
            "delay": True,
            "device": True,
        }

    def process(self, data, ctx):
        root = Path(ctx.cfg["markers"])
        state = ctx.checkpoint_state
        if state is None:
            state = {"saved": 0, "step": 0}
        for index in ctx.resume_range(0, 2):
            state["step"] = index + 1
            if index == 0:
                (root / f"{ctx.run_id}.started").write_text(str(os.getpid()))
                state["saved"] += 1
                (root / f"{ctx.run_id}.checkpoint").touch()
                ctx.checkpoint(index + 1, state)
                continue
            if ctx.cfg["crash"]:
                os._exit(7)
            for step in range(2):
                ctx.log_metrics({"train": {"loss": step}}, step=step)
            try:
                if ctx.attempt == 1:
                    deadline = monotonic() + 5
                    while (
                        ctx.cfg["wait_for_release"] and not (root / "release").exists()
                    ):
                        if monotonic() > deadline:
                            raise RuntimeError("test release timed out")
                        sleep(0.01)
                sleep(ctx.cfg["delay"])
                worker_info = {
                    "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "import_device": IMPORT_DEVICE,
                    "pid": os.getpid(),
                    "saved": state["saved"],
                    "cfg_devices": list(ctx.cfg["device"]),
                    "threads": os.environ.get("OMP_NUM_THREADS"),
                    "mirror": os.environ.get("HF_ENDPOINT"),
                }
                (root / f"{ctx.run_id}.finished").write_text(json.dumps(worker_info))
            finally:
                (root / f"{ctx.run_id}.finally").touch()
        return None


class CudaOomOnce(Stage):
    def process(self, data, ctx):
        if ctx.attempt == 1:
            raise RuntimeError("CUDA out of memory")


class SharedWork(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {}

    def process(self, data, ctx):
        ctx.log_metrics({"loss": 0.25}, step=0)
        pid = os.getpid()
        ctx.checkpoint(1, {"producer": pid})
        return pid


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
        total = data + ctx.cfg["offset"]
        ctx.log_metrics({"total": total})
        (root / f"{ctx.run_id}.consume").write_text(str(total))


class ConditionalWork(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        dependencies = {"Baseline": True, "imputer": True}
        if cfg["Baseline"] == "B2":
            dependencies["predictor"] = True
        return dependencies

    def process(self, data, ctx) -> None:
        pass


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


def _worker(**fields):
    """A scheduler worker stand-in; every field admission reads is numeric."""
    values = {
        "started": 0.0,
        "device": 0,
        "peak_kb": 0.0,
        "resident_kb": 0.0,
        "host_reserve_kb": 0.0,
        "host_limit_kb": None,
        "gpu_reserve_kb": 0.0,
        "gpu_used_kb": 0.0,
        "gpu_cap_kb": float("inf"),
        "finished": None,
        "limit": None,
    }
    values.update(fields)
    return Mock(**values)


def make_batch(root, *, count=3, devices=None, **options):
    cfg = {
        "device": [0, 1] if devices is None else devices,
        "seed": 0,
        "markers": str(root),
        "crash": False,
        "wait_for_release": False,
        "delay": 0.1,
        "item": 0,
        **options,
    }
    path = root / "experiments.yaml"
    path.write_text(yaml.safe_dump(cfg) + f"grid:\n  item: {list(range(count))}\n")
    return batch_helpers.make_batch(
        Pipeline([GpuWork]), path, output_dir=root / "batch"
    )


def warm(batch, *, gpu_process_peak_kb=1024, peak_kb=4096):
    """Supply measured per-Stage peaks so packing uses them in place of the 1 GiB defaults."""
    for experiment in batch.experiments:
        for stage_index in range(len(experiment.pipeline.stages)):
            batch.state.stage_history.append(
                {
                    "run_id": experiment.run_id,
                    "stage": stage_index,
                    "status": Status.SUCCEEDED.value,
                    "peak_kb": float(peak_kb),
                    "gpu_process_peak_kb": float(gpu_process_peak_kb),
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
        monkeypatch.setattr(settings, "poll_interval", 0.01)

    def test_a_broken_recorder_does_not_drop_the_worker_result(self, tmp_path, caplog):
        batch = make_batch(tmp_path, count=1, devices=[0])
        run = batch.experiments[0]
        ctx = RunContext(run_id=run.run_id)
        ctx.report_progress(2, total=3)
        event = ctx.recorder.events[-1]
        result = StageResult(run.run_id, 0, 1, Status.SUCCEEDED, 0.2)
        channel = Mock()
        channel.drain.return_value = [
            {"type": "event", "event": event},
            {"type": "finished", "result": result},
        ]
        worker = Worker(Mock(), run, 0, None, 0.0, channel=channel)
        recorder = Mock()
        recorder.record.side_effect = RuntimeError("sink unavailable")
        worker.read_messages(recorder)
        assert worker.progress is event and worker.finished is result
        assert run.run_id in caplog.text

    @pytest.mark.parametrize("fail_after_save", [False, True])
    def test_startup_failure_restores_queue_and_attempt_number(
        self, tmp_path, monkeypatch, fail_after_save
    ):
        batch = make_batch(tmp_path, count=1, devices=[0])
        scheduler = GpuScheduler(batch)
        run_id = batch.experiments[0].run_id
        candidate = scheduler.stages.ready_candidates(8 * 1024**2, {})[0]
        state_before = batch.store.read_manifest()

        def fail(*args):
            raise OSError("startup failure")

        if fail_after_save:

            def fail_limit(*args):
                active = batch.store.read_manifest()
                assert active["active_gpu"] == {run_id: 0}
                assert active["stage_attempts"] == {f"{run_id}:0": 1}
                assert not active["queue"]
                fail()

            monkeypatch.setattr("expman.resources.limits.memory_limit", fail_limit)
        else:
            monkeypatch.setattr("expman.scheduling.scheduler.write_record", fail)

        with pytest.raises(OSError, match="startup failure"):
            scheduler._launch(0, Memory([0]).sample()[0], Host().sample(), candidate)

        state_after = batch.store.read_manifest()
        for key in ("active_gpu", "stage_attempts", "queue"):
            assert state_after[key] == state_before[key]
        assert list(batch.state.queue) == [run_id]
        assert not batch.state.active_gpu and batch.state.next_attempt(run_id, 0) == 1
        assert not scheduler.workers

    @pytest.mark.parametrize("fail_save", [False, True])
    def test_finishing_after_a_cancellation_preserves_retry_or_recovery(
        self, tmp_path, monkeypatch, fail_save
    ):
        batch = make_batch(tmp_path, count=1, devices=[0])
        scheduler = GpuScheduler(batch)
        run = batch.experiments[0]
        state = batch.state
        state.stage_history.append(
            {"run_id": run.run_id, "stage": 0, "status": "cancelled", "attempt": 1}
        )
        state.stage_attempts[f"{run.run_id}:0"] = 1
        attempt = state.start(run.run_id, 0, 0)
        batch._save()
        worker = Worker(
            process=Mock(returncode=7),
            experiment=run,
            device=0,
            uuid="gpu",
            started=perf_counter(),
            stage_attempt=attempt,
        )
        scheduler.workers[run.run_id] = worker
        monkeypatch.setattr(
            "expman.scheduling.scheduler._kill_group", lambda _process: None
        )
        if fail_save:
            monkeypatch.setattr(
                batch.store, "save_manifest", Mock(side_effect=OSError("full"))
            )
            with pytest.raises(OSError, match="full"):
                scheduler._finish(run.run_id)
            # The previous active record remains authoritative if collection fails.
            assert batch.store.read_manifest()["active_gpu"] == {run.run_id: 0}
        else:
            scheduler._finish(run.run_id)
            assert not batch.store.read_manifest()["active_gpu"]

        resumed = batch_helpers.restore_batch(batch.pipeline, batch.output_dir)
        assert list(resumed.state.queue) == [run.run_id]
        assert resumed.state.next_attempt(run.run_id, 0) == 3
        assert resumed.results[0].status is (
            Status.CANCELLED if fail_save else Status.FAILED
        )

    @pytest.mark.simulated_gpu
    def test_worker_uses_the_parent_settings_after_importing_the_pipeline(
        self, tmp_path, monkeypatch
    ):
        module_name = "worker_settings_stage"
        (tmp_path / f"{module_name}.py").write_text(
            "import os\n"
            "from pathlib import Path\n"
            "import expman\n"
            'if "EXPMAN_IPC_FD" in os.environ:\n'
            "    expman.set(query_timeout=2.0)\n"
            "class ReadSettings(expman.Stage):\n"
            "    def process(self, data, ctx):\n"
            '        values = expman.get("query_timeout"), expman.get("retries")\n'
            '        Path(ctx.cfg["marker"]).write_text(str(values))\n'
        )
        monkeypatch.syspath_prepend(tmp_path)
        monkeypatch.setattr(settings, "retries", 4)
        try:
            stage = import_module(module_name).ReadSettings
            pipeline = Pipeline([stage])
            batch = batch_helpers.make_batch(
                pipeline,
                {"device": [0], "seed": 0, "marker": str(tmp_path / "settings")},
                output_dir=tmp_path / "batch",
            )
            # A new execution session may override its monitor timeout. The batch
            # keeps the retry count it recorded when it was created.
            monkeypatch.setattr(settings, "query_timeout", 5.0)
            monkeypatch.setattr(settings, "retries", 0)
            resumed = batch_helpers.restore_batch(pipeline, batch.output_dir)
            (result,) = resumed.run()
            assert result.status is Status.SUCCEEDED
            assert (tmp_path / "settings").read_text() == "(5.0, 4)"
        finally:
            sys.modules.pop(module_name, None)

    @pytest.mark.simulated_gpu
    def test_multiple_devices_and_metrics_results_are_collected(
        self, tmp_path, monkeypatch
    ):
        for name in (
            *environment.THREAD_VARIABLES,
            "HF_ENDPOINT",
            environment.THREADS_OVERRIDE,
            environment.MIRROR_OVERRIDE,
        ):
            monkeypatch.delenv(name, raising=False)
        batch = make_batch(tmp_path, count=2)
        results = batch.run()
        assert all(item.status is Status.SUCCEEDED for item in results)
        workers = [
            json.loads((tmp_path / f"{item.run_id}.finished").read_text())
            for item in results
        ]
        assert {item["device"] for item in workers} == {"GPU-test-0", "GPU-test-1"}
        assert len({item["pid"] for item in workers}) == 2
        assert all(item["threads"] == str(settings.worker_threads) for item in workers)
        assert all(item["mirror"] == settings.hf_mirror for item in workers)
        recorded = read_record(
            next(batch.output_dir.rglob("bootstrap.pkl")), PickleSerializer()
        )["environment"]
        assert recorded["HF_ENDPOINT"] == settings.hf_mirror
        assert recorded["OMP_NUM_THREADS"] == str(settings.worker_threads)
        assert all(item["import_device"] == item["device"] for item in workers)
        assert all(item["cfg_devices"] == [0, 1] for item in workers)
        history = batch.state.stage_history
        assert len(history) == 2
        # The two attempts ran on different GPUs during an overlapping interval.
        assert max(item["start_seconds"] for item in history) < min(
            item["end_seconds"] for item in history
        )
        expected_limit = int(cap_kb(settings.gpu_peak_kb_default) // 1024) * 1024
        assert all(item["gpu_limit_kb"] == expected_limit for item in history)
        assert all(item["peak_kb"] > 0 for item in history)
        assert batch.live.host_memory["available_ratio"] == 0.75
        assert batch.live.host_memory["swap_free_kb"] == 0.0
        assert not batch.live.host_memory["tight"]
        with sqlite3.connect(batch.output_dir / "metrics.sqlite3") as db:
            rows = db.execute(
                "SELECT run_id, step, value FROM metrics ORDER BY run_id, step"
            ).fetchall()
        assert rows == sorted(
            (item.run_id, step, step) for item in results for step in (0, 1)
        )
        assert batch.recorder.events
        resumed = batch_helpers.restore_batch(Pipeline([GpuWork]), batch.output_dir)
        assert resumed.run() == results
        assert resumed.devices == (0, 1)
        assert GpuScheduler(resumed).peaks.ceiling_kb > 0

    @pytest.mark.simulated_gpu
    def test_top_level_stages_run_independently_and_record_dependencies(self, tmp_path):
        path = tmp_path / "two-stages.yaml"
        path.write_text(
            f"device: [0]\nseed: 0\nmarkers: {tmp_path}\noffset: 10\nitem: 1\n"
            "grid:\n  item: [1, 2]\n"
        )
        batch = batch_helpers.make_batch(
            Pipeline([Prepare, Consume]), path, output_dir=tmp_path / "two-stages"
        )

        results = batch.run()

        assert all(result.status is Status.SUCCEEDED for result in results)
        assert sorted(entry["stage"] for entry in batch.state.stage_history) == [
            0,
            0,
            1,
            1,
        ]
        assert all(
            (tmp_path / f"{result.run_id}.prepare").exists() for result in results
        )
        assert [
            int((tmp_path / f"{result.run_id}.consume").read_text())
            for result in results
        ] == [11, 12]
        dependencies = {
            entry["stage"]: {path for path, _digest in entry["dependencies"]}
            for entry in batch.state.stage_history
        }
        assert dependencies[0] == {("item",), ("markers",)}
        assert dependencies[1] == {("markers",), ("offset",)}
        scheduler = GpuScheduler(batch)
        assert {
            path
            for item in batch.experiments
            for path in scheduler.stages.prefix_paths(item.run_id, 1)
        } == {("item",), ("markers",), ("offset",)}
        predicted = scheduler.stages.duration(results[0].run_id, 1)
        assert predicted is not None
        assert 0 < predicted.lower <= predicted.center <= predicted.upper

    @pytest.mark.simulated_gpu
    def test_shared_parent_survives_a_failed_child_branch(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "retries", 0)
        path = tmp_path / "branches.yaml"
        path.write_text(
            f"device: [0]\nseed: 0\nmarkers: {tmp_path}\nitem: 2\noffset: 3\n"
            "grid:\n  offset: [3, invalid]\n"
        )
        batch = batch_helpers.make_batch(
            Pipeline([Prepare, Consume]), path, output_dir=tmp_path / "batch"
        )

        success, failure = batch.run()

        assert success.status is Status.SUCCEEDED
        assert (tmp_path / f"{success.run_id}.consume").read_text() == "5"
        assert failure.status is Status.FAILED
        first, second = (run._store for run in batch.experiments)
        assert first.completed_reference((0,)) == second.completed_reference((0,))
        parent = second.completed((0,))
        assert parent is not None and parent["output"] == 2

    @pytest.mark.simulated_gpu
    def test_resume_finishes_cache_cleanup_after_the_last_success(self, tmp_path):
        pipeline = Pipeline([Prepare, Consume])
        directory = tmp_path / "batch"
        batch = batch_helpers.make_batch(
            pipeline,
            {
                "device": [0],
                "seed": 0,
                "markers": str(tmp_path),
                "item": 2,
                "offset": 3,
            },
            output_dir=directory,
        )
        with (
            patch.object(PrefixCache, "reclaim", side_effect=KeyboardInterrupt),
            pytest.raises(KeyboardInterrupt),
        ):
            batch.run()

        cache = directory / "cache"
        (saved,) = cache.glob("*/root/*/snapshot.pkl")
        # A worker can die after writing a snapshot but before linking it to a run.
        orphan = saved.parent.with_name("f" * 32)
        orphan.mkdir()
        (orphan / "snapshot.pkl").write_bytes(b"unpublished snapshot")
        resumed = batch_helpers.restore_batch(pipeline, directory)
        assert not resumed.state.queue and not resumed.state.active_gpu
        assert resumed.results[0].status is Status.SUCCEEDED
        with patch.object(GpuScheduler, "_launch") as launch:
            resumed.run()
        launch.assert_not_called()
        assert not list(cache.iterdir())

    def test_zero_sample_cannot_price_an_mps_context(self, tmp_path):
        batch = make_batch(tmp_path, count=2, devices=[0])
        scheduler = GpuScheduler(batch)
        batch.state.stage_history.append(
            {
                "run_id": batch.experiments[0].run_id,
                "stage": 0,
                "gpu_process_peak_kb": 0.0,
                "status": Status.SUCCEEDED.value,
                "capped": False,
                "gpu_capped": False,
                "dependencies": [],
            }
        )

        assert (
            scheduler.stages.peak(
                batch.experiments[1].run_id,
                0,
                "gpu_process_peak_kb",
                settings.gpu_peak_kb_default,
            )
            == settings.gpu_peak_kb_default
        )

    def test_gpu_reservations_and_ceilings_never_exceed_card_capacity(self, tmp_path):
        batch = make_batch(tmp_path, count=2, devices=[0])
        scheduler = GpuScheduler(batch)
        capacity_kb = 3 * 1024

        with patch.object(scheduler.stages, "peak", return_value=100 * capacity_kb):
            candidates = scheduler.stages.ready_candidates(
                capacity_kb, scheduler.workers
            )

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

        with patch.object(scheduler.stages, "peak", return_value=estimate):
            candidates = scheduler.stages.ready_candidates(
                8 * 1024**2, scheduler.workers
            )

        assert candidates
        assert all(item.host_reserve_kb == estimate for item in candidates)
        assert all(
            cap_kb(item.host_reserve_kb) > item.host_reserve_kb for item in candidates
        )

    def test_gpu_admission_charges_the_reserve_under_the_mps_limit(self, tmp_path):
        batch = make_batch(tmp_path, count=2, devices=[0])
        scheduler = GpuScheduler(batch)
        capacity_kb = 2 * 1024**2

        with patch.object(scheduler.stages, "peak", return_value=capacity_kb // 2):
            candidates = scheduler.stages.ready_candidates(
                capacity_kb, scheduler.workers
            )

        assert candidates
        assert all(item.gpu_reserve_kb == capacity_kb // 2 for item in candidates)
        assert all(
            item.gpu_cap_kb == capacity_kb // 2 * settings.capacity_factor
            for item in candidates
        )

    def test_a_running_worker_is_charged_up_to_its_reservation(self, tmp_path):
        batch = make_batch(tmp_path, count=1, devices=[0], delay=0)
        scheduler = GpuScheduler(batch)
        scheduler.workers["running"] = Mock(
            started=0.0, peak_kb=0.0, resident_kb=0.0, host_reserve_kb=2 * 1024**2
        )
        host = HostMemory(64 * 1024**2, settings.host_reserve_kb + 3 * 1024**2, 0, 0)

        assert _admits(scheduler.workers, host, 1024**2)
        assert not _admits(scheduler.workers, host, 2 * 1024**2)

    def test_known_stage_group_has_only_one_runnable_representative(self, tmp_path):
        path = tmp_path / "groups.yaml"
        path.write_text(
            f"device: [0]\nseed: 0\nmarkers: {tmp_path}\noffset: 0\nunused: 1\n"
            "grid:\n  unused: [1, 2, 3]\n"
        )
        batch = batch_helpers.make_batch(
            Pipeline([SharedWork, Consume]), path, output_dir=tmp_path / "groups"
        )
        scheduler = GpuScheduler(batch)

        candidates = scheduler.stages.ready_candidates(16 * 1024**2, scheduler.workers)

        assert len(candidates) == 1

    def test_stage_shortlist_takes_retries_then_queue_order(
        self, tmp_path, monkeypatch
    ):
        batch = make_batch(tmp_path, count=5, devices=[0])
        state = batch.state
        completed, retry, first, second, _ = state.queue
        for run_id, status in (
            (completed, Status.SUCCEEDED),
            (retry, Status.CANCELLED),
        ):
            state.start(run_id, 0, 0)
            state.release(run_id)
            state.stage_history.append(
                {"run_id": run_id, "stage": 0, "status": status.value}
            )
        state.retry(retry)
        monkeypatch.setattr(settings, "candidates_per_stage", 3)

        candidates = GpuScheduler(batch).stages.ready_candidates(16 * 1024**2, {})

        assert [candidate.run_id for candidate in candidates] == [retry, first, second]

    def test_conditional_dependencies_follow_the_config_value(self, tmp_path):
        path = tmp_path / "conditional.yaml"
        path.write_text(
            "device: [0]\nseed: 0\n"
            "Baseline: B1\nimputer: i\npredictor: p1\n"
            "grid:\n"
            "  Baseline: [B1, B2]\n  predictor: [p1, p2]\n"
        )
        batch = batch_helpers.make_batch(
            Pipeline([ConditionalWork]), path, output_dir=tmp_path / "conditional"
        )
        scheduler = GpuScheduler(batch)
        groups = {
            (experiment._cfg["Baseline"], experiment._cfg["predictor"]): (
                scheduler.stages.group(experiment.run_id, 0)
            )
            for experiment in batch.experiments
        }

        # Both branches depend on Baseline and imputer. Only B2 uses predictor.
        assert groups[("B1", "p1")] == groups[("B1", "p2")]
        assert groups[("B2", "p1")] != groups[("B2", "p2")]
        assert groups[("B1", "p1")] != groups[("B2", "p1")]

    def test_stage_groups_preserve_each_parents_conditional_declaration(self, tmp_path):
        class Select(Stage):
            @classmethod
            def config_dependencies(cls, cfg):
                return {cfg["which"]: True}

        class Combine(Stage):
            @classmethod
            def config_dependencies(cls, cfg):
                return {"left": True, "right": True}

            def process(self, data, ctx) -> None:
                pass

        path = tmp_path / "cfg.yaml"
        path.write_text(
            "device: [0]\nseed: 0\nleft: 1\nright: 2\nwhich: left\ngrid:\n  which: [left, right]\n"
        )
        batch = batch_helpers.make_batch(
            Pipeline([Select, Combine]), path, output_dir=tmp_path / "batch"
        )
        scheduler = GpuScheduler(batch)
        first, second = batch.by_id
        assert scheduler.stages.prefix_paths(first, 1) == scheduler.stages.prefix_paths(
            second, 1
        )
        assert scheduler.stages.group(first, 1) != scheduler.stages.group(second, 1)
        batch.state.stage_progress.update({first: 1, second: 1})
        assert len(scheduler.stages.ready_candidates(16 * 1024**2, {})) == 2
        assert batch.estimate().remaining_experiments == 2

    @pytest.mark.parametrize("allocated", [False, True])
    def test_later_gpu_work_uses_an_idle_card(self, tmp_path, allocated):
        batch = make_batch(tmp_path, count=1, devices=[1, 2, 3, 4])
        scheduler = GpuScheduler(batch)
        run_id = batch.experiments[0].run_id
        request = 10 * 1024**2
        candidate = Candidate(run_id, 0, 1024, request, request, None)
        memory = {
            device: DeviceMemory(f"GPU-{device}", 50 * 1024, 50 * 1024)
            for device in batch.devices
        }
        with patch.object(
            scheduler.stages, "ready_candidates", return_value=[candidate]
        ):
            first = scheduler._launch_plan(memory, Host().sample())
            assert [device for _, device in first] == [1]
            # A later tick must account for both actual usage and promises that
            # an existing worker has not allocated yet.
            scheduler.workers["previous"] = _worker(
                device=1,
                gpu_reserve_kb=request,
                gpu_used_kb=request if allocated else 0.0,
            )
            if allocated:
                memory[1] = DeviceMemory("GPU-1", 50 * 1024, 40 * 1024)
            second = scheduler._launch_plan(memory, Host().sample())
        assert [device for _, device in second] == [2]

    @pytest.mark.parametrize("host_gib,gpu_gib", [(16, 8), (503, 48)])
    def test_initial_budgets_are_one_gib_independent_of_machine_size(
        self, tmp_path, host_gib, gpu_gib
    ):
        batch = make_batch(tmp_path, count=3, devices=[0, 1, 2])
        scheduler = GpuScheduler(batch)
        memory = {
            device: DeviceMemory(f"GPU-{device}", gpu_gib * 1024, gpu_gib * 1024)
            for device in batch.devices
        }
        host = HostMemory(host_gib * 1024**2, host_gib * 1024**2 // 2, 0, 0)
        plan = scheduler._launch_plan(memory, host)
        assert len(plan) == 3
        assert {device for _, device in plan} == {0, 1, 2}
        assert all(c.host_reserve_kb == c.gpu_reserve_kb == 1024**2 for c, _ in plan)
        tight = HostMemory(host.total_kb, settings.host_reserve_kb + 512 * 1024, 0, 0)
        assert not scheduler._launch_plan(memory, tight)

    @pytest.mark.simulated_gpu
    def test_mps_service_honours_tmpdir(self, tmp_path, monkeypatch):
        from expman.resources.mps import MpsService

        monkeypatch.setenv("TMPDIR", str(tmp_path))
        service = MpsService()
        try:
            deadline = monotonic() + 10
            while not service.ready(["GPU-test-0"]):
                assert monotonic() < deadline
                sleep(0.01)
            assert service.pipe is not None
            directory = Path(service.pipe)
            assert directory.parent == tmp_path and directory.is_dir()
        finally:
            service.close()
        assert not directory.exists()

    def test_gpu_plan_respects_available_memory_after_balancing(self, tmp_path):
        batch = make_batch(tmp_path, count=1, devices=[1, 2])
        scheduler = GpuScheduler(batch)
        request = 10 * 1024**2
        candidate = Candidate(
            batch.experiments[0].run_id, 0, 1024, request, request, None
        )
        memory = {
            1: DeviceMemory("GPU-1", 50 * 1024, 9 * 1024),
            2: DeviceMemory("GPU-2", 50 * 1024, 10 * 1024),
        }
        with patch.object(
            scheduler.stages, "ready_candidates", return_value=[candidate]
        ):
            plan = scheduler._launch_plan(memory, Host().sample())
        assert [device for _, device in plan] == [2]

    def test_later_candidate_that_fills_the_card_wins_even_by_one_mib(self, tmp_path):
        batch = make_batch(tmp_path, count=2, devices=[0])
        first, second = batch.state.queue
        scheduler = GpuScheduler(batch)
        memory = {0: DeviceMemory("GPU-0", 2048, 2048)}

        def peak(run_id, stage_index, field, default):
            if field == "gpu_process_peak_kb":
                return (2047 if run_id == first else 2048) * 1024
            return default

        with patch.object(scheduler.stages, "peak", side_effect=peak):
            ((candidate, device),) = scheduler._launch_plan(memory, Host().sample())

        assert (candidate.run_id, device) == (second, 0)

    @pytest.mark.parametrize(
        ("estimate_gib", "large_free_gib", "expected_device", "expected_cap_gib"),
        [(10, 24, 1, 11), (4, 0, 0, 4)],
    )
    def test_gpu_plan_prices_work_for_the_card_that_will_run_it(
        self, tmp_path, estimate_gib, large_free_gib, expected_device, expected_cap_gib
    ):
        scheduler = GpuScheduler(make_batch(tmp_path, count=1, devices=[0, 1]))
        memory = {
            0: DeviceMemory("GPU-0", 4 * 1024, 4 * 1024),
            1: DeviceMemory("GPU-1", 24 * 1024, large_free_gib * 1024),
        }
        with patch.object(
            scheduler.stages, "peak", return_value=estimate_gib * 1024**2
        ):
            ((candidate, device),) = scheduler._launch_plan(memory, Host().sample())
        assert device == expected_device
        assert candidate.gpu_reserve_kb == estimate_gib * 1024**2
        assert candidate.gpu_cap_kb == pytest.approx(expected_cap_gib * 1024**2)

    def test_requested_host_limit_is_not_an_enforced_limit(self):
        worker = _worker(limit=MemoryLimit((), None, 100.0, "systemd"))
        assert not worker.limit.capped
        Worker.update_resources(
            worker,
            {
                "host_current_kb": 20.0,
                "host_peak_kb": 30.0,
                "host_usage_source": "proc",
                "host_usage_error": "per-attempt memory controller unavailable",
                "host_limit_kb": None,
                "host_limit_verified": False,
                "cgroup": None,
            },
        )
        assert worker.resident_kb == 20.0
        assert worker.peak_kb == 30.0
        assert worker.host_usage_source == "proc"
        assert not worker.limit.capped
        assert worker.limit.cap_kb == 100.0

        Worker.update_resources(
            worker,
            {
                "host_current_kb": 25.0,
                "host_peak_kb": 25.0,
                "host_limit_kb": 99.0,
                "host_limit_verified": True,
                "cgroup": "/memory/expman-worker.scope",
            },
        )
        assert worker.limit.capped
        assert worker.host_limit_kb == 99.0
        assert worker.peak_kb == 30.0
        assert worker.limit.cap_kb == 100.0
        # A later unreadable controller must not leave a stale enforced badge.
        Worker.update_resources(
            worker,
            {"host_limit_kb": None, "host_limit_verified": False, "cgroup": None},
        )
        assert not worker.limit.capped
        assert worker.limit.cgroup is None
        assert worker.limit.cap_kb == 100.0

    @pytest.mark.parametrize("cap", [0.0, 105.0, float("inf"), float("nan")])
    def test_invalid_host_limit_report_cannot_verify_a_ceiling(self, cap):
        worker = _worker(limit=MemoryLimit((), None, 100.0, "systemd"))
        Worker.update_resources(
            worker, {"host_limit_kb": cap, "host_limit_verified": True}
        )
        assert not worker.limit.capped

    def test_tick_greedily_fills_a_card(self, tmp_path):
        batch = warm(make_batch(tmp_path, count=4, devices=[0]))
        scheduler = GpuScheduler(batch)
        started = []

        def launch(device, memory, host, candidate):
            started.append(device)
            batch.state.queue.remove(candidate.run_id)
            return True

        with patch.object(scheduler, "_launch", side_effect=launch):
            scheduler._launch_available(Memory([0]).sample(), Host().sample())

        assert started == [0, 0, 0, 0]

    def test_tick_plan_combines_memory_and_gpu_stages(self, tmp_path):
        path = tmp_path / "two-stage.yaml"
        path.write_text(
            f"device: [0]\nseed: 0\nmarkers: {tmp_path}\n"
            "crash: false\nwait_for_release: false\ndelay: 0\n"
            "item: 0\ngrid:\n  item: [0, 1, 2, 3, 4, 5, 6, 7]\n"
        )
        batch = batch_helpers.make_batch(
            Pipeline([GpuWork, GpuWork]), path, output_dir=tmp_path / "plan"
        )
        for experiment in batch.experiments[4:]:
            batch.state.stage_progress[experiment.run_id] = 1
        scheduler = GpuScheduler(batch)

        def peak(run_id, stage, field, default):
            if field == "peak_kb":
                return (512 if stage == 0 else 1024) * 1024
            return 0.0 if stage == 0 else 1024 * 1024

        host = HostMemory(64 * 1024**2, settings.host_reserve_kb + 5 * 1024**2, 0, 0)
        memory = {0: DeviceMemory("GPU-test-0", 3000, 3000)}
        with patch.object(scheduler.stages, "peak", side_effect=peak):
            plan = scheduler._launch_plan(memory, host)

        stages = [candidate.stage_index for candidate, _device in plan]
        # One joint plan, not a barrier per Stage: both Stages share this tick.
        assert set(stages) == {0, 1}
        # The card holds 3000 MiB, so a third 1 GiB Stage 1 reservation does not
        # fit and every remaining slot goes to the Stage that reserves none.
        assert stages.count(0) == 4
        assert stages.count(1) == 2

    @pytest.mark.simulated_gpu
    def test_shared_prefix_is_materialized_without_worker_per_member(self, tmp_path):
        output_dir = tmp_path / "shared-two"
        path = tmp_path / "shared.yaml"
        path.write_text(
            f"device: [0]\nseed: 0\nmarkers: {tmp_path}\noffset: 1\n"
            "combines: [{unused: 1}, {unused: 2}, {offset: 2}]\n"
        )
        pipeline = Pipeline([SharedWork, Consume])
        batch = batch_helpers.make_batch(pipeline, path, output_dir=output_dir)
        seeded = Experiment(
            pipeline,
            {"seed": 0},
            output_dir=tmp_path / "seed",
            _cache_root=output_dir / "cache",
        )
        assert seeded.run_stage(0, attempt=1).status is Status.SUCCEEDED
        prefix = seeded._store.completed((0,))
        assert prefix is not None
        expected = prefix["output"]

        results = batch.run()
        assert all(item.status is Status.SUCCEEDED for item in results)
        assert [
            entry["reused"]
            for entry in batch.state.stage_history
            if entry["stage"] == 0
        ] == [True, True, True]
        assert len(list(tmp_path.glob("*.consume"))) == 2
        with sqlite3.connect(output_dir / "metrics.sqlite3") as db:
            for result, offset in zip(results, [1, 1, 2], strict=True):
                metrics = dict(
                    db.execute(
                        "SELECT metric_path, value FROM metrics WHERE run_id = ?",
                        (result.run_id,),
                    )
                )
                assert metrics == {'["loss"]': 0.25, '["total"]': expected + offset}
        assert not list((output_dir / "cache").rglob("*.pkl"))

    @pytest.mark.simulated_gpu
    def test_unexpected_process_exit_exhausts_one_retry(self, tmp_path):
        batch = make_batch(tmp_path, count=1, devices=[0], crash=True)
        (result,) = batch.run()
        assert result.status is Status.FAILED
        assert [attempt.error_type for attempt in result.attempts] == [
            "WorkerExit",
            "WorkerExit",
        ]

    @pytest.mark.simulated_gpu
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
            batch.run()
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
        resumed = batch_helpers.restore_batch(Pipeline([GpuWork]), batch.output_dir)
        results = resumed.run()
        assert all(item.status is Status.SUCCEEDED for item in results)
        # The state attribute comes from the checkpoint, not a re-executed loop.
        saved = [
            json.loads((tmp_path / f"{item.run_id}.finished").read_text())["saved"]
            for item in results
        ]
        assert len(saved) == 2
        assert all(value == 1 for value in saved)
        assert all(len(item.attempts) in (1, 2) for item in results)
        assert max(len(item.attempts) for item in results) == 2
        assert max(resumed.state.stage_attempts.values()) == 2

    def test_host_pressure_does_not_shed_the_only_running_attempt(self, tmp_path):
        # Shedding distributes a shared host between attempts. With one attempt
        # there is nothing to distribute, and ending it discards work no other
        # attempt is competing with while the worker's cgroup is what bounds
        # the host. A shortage with one worker pauses launches instead.
        batch = make_batch(tmp_path, count=1, devices=[0], delay=0)
        scheduler = GpuScheduler(batch)
        short = HostMemory(1000, 5, 0, 0)
        shed = []
        with patch.object(
            GpuScheduler, "_shed", side_effect=lambda *a: shed.append(a[0])
        ):
            scheduler.workers["only"] = _worker(started=0.0)
            scheduler._relieve(None, short)
            assert shed == []

            # A second attempt is over-commitment, so the newest one goes.
            scheduler.workers["newer"] = _worker(started=1.0)
            scheduler._relieve(None, short)
            assert shed == ["newer"]

            # An unreadable host is the same shortage, not a reason to stop the
            # one attempt that is already running.
            scheduler.workers.pop("newer")
            scheduler._relieve(None, None)
            assert shed == ["newer"]

    @pytest.mark.simulated_gpu
    def test_cuda_oom_takes_priority_over_nearby_host_threshold(self, tmp_path):
        batch = batch_helpers.make_batch(
            Pipeline([CudaOomOnce]),
            {"device": [0], "seed": 0},
            output_dir=tmp_path / "cuda-oom",
        )
        with patch.object(GpuScheduler, "_cap_report", return_value=(True, 8192)):
            result = batch.run()[0]
        assert result.status is Status.SUCCEEDED
        first = batch.state.stage_history[0]
        assert first["gpu_capped"] and not first["capped"]
        assert first["gpu_process_peak_kb"] == 0
        assert first["gpu_retry_floor_kb"] > 0

    def test_nearing_unverified_host_limit_does_not_mask_a_worker_crash(self, tmp_path):
        scheduler = GpuScheduler(make_batch(tmp_path, count=1))
        worker = _worker(
            limit=MemoryLimit((), None, 100.0, "none"),
            peak_kb=99.0,
            memory_events={},
            cap_hit=False,
        )
        worker.process.returncode = 7
        assert scheduler._cap_report(worker) == (False, 99.0)

    @pytest.mark.parametrize(("peak", "refusal"), [(95.0, {}), (0.0, {"oom_kill": 1})])
    def test_host_cap_retry_uses_kernel_limit_even_if_usage_sample_was_missed(
        self, tmp_path, peak, refusal
    ):
        batch = make_batch(tmp_path, count=1)
        scheduler = GpuScheduler(batch)
        run_id = batch.state.queue[0]
        batch.state.start(run_id, 0, 0)
        worker = Worker(
            Mock(returncode=7),
            batch.by_id[run_id],
            0,
            "GPU-test-0",
            perf_counter(),
            limit=MemoryLimit((), None, 200.0, "systemd", verified=True),
            host_limit_kb=100.0,
            peak_kb=peak,
            memory_events=refusal,
        )
        scheduler.workers[run_id] = worker
        with patch("expman.scheduling.scheduler._kill_group"):
            scheduler._finish(run_id)
        row = batch.state.stage_history[-1]
        assert row["status"] == Status.CANCELLED.value
        assert row["peak_kb"] == peak
        assert row["host_retry_floor_kb"] == 100.0
        assert scheduler.stages.peak(run_id, 0, "peak_kb", 50.0) > 100.0

    def test_sampling_above_cap_does_not_kill_an_mps_client(self, tmp_path):
        scheduler = GpuScheduler(make_batch(tmp_path, count=1))
        scheduler.workers["owner"] = _worker(gpu_used_kb=2048, gpu_cap_kb=1024)
        with patch.object(scheduler, "_finish") as finish:
            scheduler._relieve(Memory([0]).sample(), Host().sample())
        finish.assert_not_called()

    def test_unknown_gpu_usage_blocks_only_that_card(self, tmp_path):
        scheduler = GpuScheduler(make_batch(tmp_path, count=1))
        scheduler.workers["unknown"] = _worker(
            device=0, stage_index=-1, gpu_used_kb=None
        )
        placements = scheduler._launch_plan(Memory([0, 1]).sample(), Host().sample())
        assert [device for _, device in placements] == [1]
        assert scheduler.last_plan["capacity"]["gpu_left_kb"][0] == 0
        assert not scheduler._launch_plan(Memory([0]).sample(), Host().sample())
        assert not scheduler.last_plan["placements"]

    def test_allocator_peak_does_not_price_process_memory(self, tmp_path):
        batch = make_batch(tmp_path, count=1)
        rid = batch.experiments[0].run_id
        batch.state.stage_history.append(
            {
                "run_id": rid,
                "stage": 0,
                "gpu_peak_kb": 0,
                "status": Status.SUCCEEDED.value,
            }
        )
        scheduler = GpuScheduler(batch)
        assert scheduler.stages.peak(rid, 0, "gpu_process_peak_kb", 8192) == 8192

    def test_an_unbacked_device_reservation_sheds_the_newest_attempt(self, tmp_path):
        # Admission re-reads the cards on every tick, so a process that claims
        # memory after a launch is visible -- but only to the next launch.
        # Nothing re-checked the contracts already running, so a card left
        # without room throttled every attempt on it instead of shedding one.
        batch = make_batch(tmp_path, count=1, devices=[0], delay=0)
        scheduler = GpuScheduler(batch)
        memory = {0: DeviceMemory("GPU-test-0", 8 * 1024, 0.5 * 1024)}
        scheduler.workers["older"] = _worker(
            started=0.0,
            device=0,
            gpu_reserve_kb=4 * 1024**2,
            gpu_used_kb=1 * 1024**2,
        )
        scheduler.workers["newer"] = _worker(
            started=1.0, device=0, gpu_reserve_kb=4 * 1024**2, gpu_used_kb=0.0
        )
        shed = []
        with patch.object(
            GpuScheduler, "_shed", side_effect=lambda *a: shed.append(a[0])
        ):
            scheduler._relieve(memory, Host().sample())
        # The two together were promised 8 GiB on a card with 0.5 GiB left.
        assert shed == ["newer"]

    def test_attempts_inside_their_reservation_are_not_shed(self, tmp_path):
        batch = make_batch(tmp_path, count=1, devices=[0], delay=0)
        scheduler = GpuScheduler(batch)
        memory = {0: DeviceMemory("GPU-test-0", 8 * 1024, 6 * 1024)}
        scheduler.workers["older"] = _worker(
            started=0.0,
            device=0,
            gpu_reserve_kb=1 * 1024**2,
            gpu_used_kb=1 * 1024**2,
        )
        scheduler.workers["newer"] = _worker(
            started=1.0,
            device=0,
            gpu_reserve_kb=1 * 1024**2,
            gpu_used_kb=1 * 1024**2,
        )
        shed = []
        with patch.object(
            GpuScheduler, "_shed", side_effect=lambda *a: shed.append(a[0])
        ):
            scheduler._relieve(memory, Host().sample())
        assert shed == []

    def test_an_attempt_beyond_its_reservation_is_shed(self, tmp_path):
        # An attempt that took more than it was promised is spending another
        # attempt's allowance, so the contract on that card is not the one
        # admission made, whatever the reading says.
        batch = make_batch(tmp_path, count=1, devices=[0], delay=0)
        scheduler = GpuScheduler(batch)
        memory = {0: DeviceMemory("GPU-test-0", 8 * 1024, 0.5 * 1024)}
        scheduler.workers["older"] = _worker(
            started=0.0,
            device=0,
            gpu_reserve_kb=1 * 1024**2,
            gpu_used_kb=7 * 1024**2,
        )
        scheduler.workers["newer"] = _worker(
            started=1.0, device=0, gpu_reserve_kb=1 * 1024**2, gpu_used_kb=0.0
        )
        shed = []
        with patch.object(
            GpuScheduler, "_shed", side_effect=lambda *a: shed.append(a[0])
        ):
            scheduler._relieve(memory, Host().sample())
        assert shed == ["newer"]

    def test_shedding_is_paced_while_pressure_outlives_one_tick(self, tmp_path):
        # A cancelled attempt is retried, so a shortage that outlives one tick
        # must not spend one attempt per tick while the ended process's memory
        # is still being released.
        batch = make_batch(tmp_path, count=1, devices=[0], delay=0)
        scheduler = GpuScheduler(batch)
        short = HostMemory(1000, 5, 0, 0)
        for run_id, started in (("older", 0.0), ("middle", 1.0), ("newer", 2.0)):
            scheduler.workers[run_id] = _worker(started=started)
        shed = []
        with patch.object(
            GpuScheduler, "_shed", side_effect=lambda *a: shed.append(a[0])
        ):
            scheduler._relieve(None, short)
            scheduler._relieve(None, short)
            scheduler._relieve(None, short)
        assert shed == ["newer"]

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
        # Fixed reservations allow all four probes beside the live attempt.
        assert len(launched) == 4

    @pytest.mark.simulated_gpu
    def test_host_memory_shortage_pauses_launches_until_it_recovers(self, tmp_path):
        batch = make_batch(tmp_path, count=1, devices=[0], delay=0.05)
        original = Host.sample
        pressured = []

        def shortage(monitor):
            if len(pressured) < 5:
                pressured.append(len(batch.state.active_gpu))
                return HostMemory(1000, 5, 0, 0)
            return original(monitor)

        with patch.object(Host, "sample", shortage):
            results = batch.run()
        assert pressured == [0] * 5
        assert results[0].status is Status.SUCCEEDED

    @pytest.mark.simulated_gpu
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
            active = list(batch.state.active_gpu)
            if (
                not dropped
                and len(active) == 2
                and all((tmp_path / f"{key}.checkpoint").exists() for key in active)
            ):
                dropped.append(active[-1])
                cards.extend(batch.state.active_gpu.values())
                return HostMemory(1000, 5, 0, 0)
            if dropped and not (tmp_path / "release").exists():
                held.append(len(batch.state.active_gpu))
                if len(held) >= 5:
                    (tmp_path / "release").touch()
            return original(monitor)

        with (
            patch.object(settings, "host_reserve_kb", 2 * 1024**2),
            patch.object(Host, "sample", pressure),
        ):
            results = batch.run()
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

    @pytest.mark.simulated_gpu
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
            if failures < 2 and batch.state.active_gpu:
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
            results = batch.run()
        assert failures == 2
        shed = [
            attempt
            for result in results
            for attempt in result.attempts
            if attempt.error_type == "MemoryPressure"
        ]
        assert 1 <= len(shed) <= 2
        assert all(item.status is Status.SUCCEEDED for item in results)

    @pytest.mark.simulated_gpu
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
            results = batch.run()
        assert failures == 3
        assert launches
        assert all(item.status is Status.SUCCEEDED for item in results)
        assert all(len(item.attempts) == 1 for item in results)
        assert not batch.state.active_gpu


class TestDevice:
    def test_device_and_seed_are_required_before_creating_output(self, tmp_path):
        root = tmp_path
        with pytest.raises(ConfigError, match="device is required"):
            batch_helpers.make_batch(
                Pipeline([]), {"seed": 0}, output_dir=root / "no-device"
            )
        with pytest.raises(ValueError, match="seed is required"):
            batch_helpers.make_batch(
                Pipeline([]), {"device": [0]}, output_dir=root / "no-seed"
            )
        with pytest.raises(ValueError, match="seed is required"):
            Experiment(Pipeline([]), {}, output_dir=root / "experiment-no-seed")
        with pytest.raises(ConfigError, match="nonempty"):
            batch_helpers.make_batch(
                Pipeline([]),
                {"device": [], "seed": 0},
                output_dir=root / "empty-device",
            )
        assert not (root / "no-device").exists()
        assert not (root / "no-seed").exists()
        assert not (root / "experiment-no-seed").exists()
        assert not (root / "empty-device").exists()

    @pytest.mark.parametrize("value", [0, [True], [-1], [0, 0]])
    def test_invalid_device_values_are_rejected(self, tmp_path, value):
        with pytest.raises(ConfigError):
            batch_helpers.make_batch(
                Pipeline([]),
                {"device": value, "seed": 0},
                output_dir=tmp_path / "unused",
            )

    def test_fits_reserve_uses_available_bytes(self):
        nearly_full = HostMemory(
            settings.host_reserve_kb, settings.host_reserve_kb - 1, 0, 0
        )
        assert nearly_full.available_ratio > 0.9
        assert not fits_reserve(nearly_full, 0)
        available = settings.host_reserve_kb + 2 * 1024**2
        # Fully occupied swap still permits a launch with enough available RAM.
        roomy = HostMemory(64 * 1024**2, available, 4096, 0)
        assert roomy.headroom_kb == 2 * 1024**2
        assert fits_reserve(roomy, 2 * 1024**2)
        assert not fits_reserve(roomy, 2 * 1024**2 + 1)
        assert not fits_reserve(None, 0)

    def test_query_timeout_is_an_observation_error_with_the_default_timeout(self):
        with (
            patch(
                "expman.resources.devices.subprocess.run",
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
            patch("expman.resources.devices.shutil.which", return_value=None),
            patch("expman.resources.devices.NVIDIA_SMI_FALLBACKS", (fallback,)),
            patch("expman.resources.devices.subprocess.run") as query,
        ):
            query.return_value.stdout = "0, GPU-first, 100, 10, 0\n"
            NvidiaMemory((0,)).sample()
        assert query.call_args.args[0][0] == str(fallback)
        with patch(
            "expman.resources.devices.shutil.which", return_value="/usr/bin/nvidia-smi"
        ):
            assert nvidia_smi() == "/usr/bin/nvidia-smi"

    def test_nvidia_memory_validates_observations_and_identities(self):
        with patch("expman.resources.devices.subprocess.run") as query:
            query.return_value.stdout = (
                "0, GPU-first, 100, 10, 72\n1, GPU-second, 200, 150, [N/A]\n"
            )
            monitor = NvidiaMemory((0, 1))
            cards = monitor.sample()
            assert cards[1].utilization_percent is None
            observation = cards[0]
            assert observation.utilization_percent == 72
            assert observation.free / observation.total == 0.1
            assert "--id=0,1" in query.call_args.args[0]
            query.return_value.stdout = (
                "0, GPU-replaced, 100, 10, 72\n1, GPU-second, 200, 150, 0\n"
            )
            with pytest.raises(RuntimeError, match="identities changed"):
                monitor.sample()
            query.return_value.stdout = "0, GPU-first, N/A, N/A, 0\n"
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
        assert observation.tight
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

    def test_host_cpu_utilization_uses_aggregate_counter_deltas(self, tmp_path):
        meminfo = tmp_path / "meminfo"
        meminfo.write_text("MemTotal: 1000 kB\nMemAvailable: 500 kB\n")
        monitor = MeminfoMonitor(meminfo)
        stat = tmp_path / "stat"
        stat.write_text("cpu  10 0 0 20 0 0 0 0 4 0\n")
        assert monitor.sample().utilization_percent is None
        # Ten busy ticks, six idle and four waiting for I/O. Guest ticks
        # already contribute to user time and do not count a second time.
        stat.write_text("cpu  20 0 0 26 4 0 0 0 6 0\n")
        assert monitor.sample().utilization_percent == 50.0
        stat.unlink()
        observation = monitor.sample()
        assert observation.utilization_percent is None
        assert observation.available_kb == 500
