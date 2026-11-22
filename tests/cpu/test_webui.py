import json
import os
import socket
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from threading import Barrier
from time import perf_counter, time
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import batch_helpers
import pytest

from expman import Pipeline, Stage, Status, devices
from expman.cli import main
from expman.core.context import RunContext
from expman.core.events import ProgressEvent
from expman.estimation.duration_model import StageDuration
from expman.observation.archive import read_batch
from expman.observation.live import LIVE_FILE, LivePublisher, live_enabled
from expman.observation.registry import ensure_ids
from expman.observation.snapshot import SCHEMA, build_snapshot, stage_samples
from expman.process_control import STOP_FIFO, stop_on_sigterm
from expman.resources.devices import DeviceMemory, HostMemory
from expman.scheduling.parallel_estimation import estimate_parallel
from expman.scheduling.scheduler import GpuScheduler, Worker
from expman.settings import settings
from expman.storage.base import PickleSerializer, run_lock, write_record
from expman.storage.batch import BatchStore
from expman.storage.metrics import MetricStore
from expman.webui.result_layout import layout_results
from expman.webui.server import Stream, Workspace, serve

POSIX = pytest.mark.skipif(os.name != "posix", reason="POSIX worker process groups")
GIB = 1024 * 1024


class Work(Stage):
    def process(self, data, ctx):
        ctx.log_metrics({"item": ctx.cfg["item"]})


class Memory:
    """A 24 GiB card with 16 GiB free, reported in the units nvidia-smi uses."""

    def __init__(self, devices):
        self.devices = devices

    def sample(self):
        return {
            device: DeviceMemory(f"GPU-test-{device}", 24 * 1024, 16 * 1024, 72)
            for device in self.devices
        }


class Host:
    def sample(self):
        return HostMemory(64 * GIB, 40 * GIB, 0, 0, 31.5)


class Broken:
    """A stand-in for a batch whose attributes cannot be read at all."""

    def __init__(self, output_dir):
        self.output_dir = output_dir


@pytest.fixture(autouse=True)
def fake_devices(monkeypatch):
    monkeypatch.setattr(devices, "NvidiaMemory", Memory)
    monkeypatch.setattr(devices, "MeminfoMonitor", Host)
    monkeypatch.setattr(settings, "poll_interval", 0.01)


def make_batch(root, *, count=2):
    path = root / "experiments.yaml"
    path.write_text(
        f"device: [0]\nseed: 0\nitem: 0\ngrid:\n  item: {list(range(count))}\n"
    )
    return batch_helpers.make_batch(Pipeline([Work]), path, output_dir=root / "batch")


def seed_finished_first(batch, *, duration=2.0):
    """Finish the first configuration so the rest has measured samples.

    The remaining configurations stay queued, which is the state a dashboard is
    normally opened in: work left to do, with a measured Stage behind it.
    """
    experiment = batch.experiments[0]
    for stage_index in range(len(experiment.pipeline.stages)):
        batch.state.stage_history.append(
            {
                "run_id": experiment.run_id,
                "stage": stage_index,
                "attempt": 1,
                "status": Status.SUCCEEDED.value,
                "duration": duration,
                "peak_kb": 4096.0,
                "gpu_peak_kb": 1024.0,
                "capped": False,
                "gpu_capped": False,
                "reused": False,
            }
        )
    batch.state.stage_progress[experiment.run_id] = len(experiment.pipeline.stages)
    batch.state.queue.remove(experiment.run_id)
    return batch


def live_worker(batch, run_id, *, device=0, elapsed=30.0):
    """Install one running attempt with the readings a scheduler tick would leave."""
    experiment = next(item for item in batch.experiments if item.run_id == run_id)
    batch._scheduler.workers[run_id] = Worker(
        process=Mock(),
        experiment=experiment,
        device=device,
        uuid=f"GPU-test-{device}",
        started=perf_counter() - elapsed,
        stage_index=0,
        stage_attempt=2,
        host_reserve_kb=8 * GIB,
        gpu_reserve_kb=6 * GIB,
        gpu_cap_kb=7 * GIB,
        mps_client=(1, 2),
        expected=StageDuration(center=90.0, lower=60.0, upper=150.0),
        resident_kb=3 * GIB,
        peak_kb=4 * GIB,
        gpu_used_kb=2 * GIB,
        gpu_process_peak_kb=3 * GIB,
    )
    batch.state.active_gpu[run_id] = device
    batch.live.device_memory = {
        device: {
            "uuid": f"GPU-test-{device}",
            "total_kb": 24 * GIB,
            "free_kb": 16 * GIB,
            "observed_at": time(),
        }
    }
    batch.live.host_memory = {
        "available_ratio": 0.625,
        "available_kb": 40 * GIB,
        "total_kb": 64 * GIB,
        "headroom_kb": 39 * GIB,
        "swap_total_kb": 0.0,
        "swap_free_kb": 0.0,
        "tight": False,
        "admits_next": True,
        "observed_at": time(),
    }


def write_batch_dir(root, name, *, runs=2, status="succeeded", stages=1):
    """A batch directory written straight to disk, without running anything."""
    directory = root / name
    directory.mkdir(parents=True)
    experiments = [
        {
            "run_id": f"{index:032x}",
            "attempts": [
                {
                    "attempt": 1,
                    "status": status,
                    "duration_seconds": 1.0,
                    "error_type": None,
                    "error_message": None,
                }
            ],
        }
        for index in range(runs)
    ]
    pending = status == "pending"
    manifest = {
        "pipeline": [
            {"class": f"demo.Stage{index}", "source": None} for index in range(stages)
        ],
        "pipeline_spec": "demo:train",
        "max_retries": 1,
        "experiments": experiments,
        "queue": [entry["run_id"] for entry in experiments] if pending else [],
        "active_gpu": {},
        "stage_progress": {},
        "stage_attempts": {},
        "stage_elapsed": {},
        "stage_history": [],
        "estimate_coverage": 0.8,
    }
    serializer = PickleSerializer()
    BatchStore(directory, serializer).save_manifest(manifest)
    write_record(
        directory / "timing.pkl",
        {"version": 1, "elapsed_seconds": 120.0, "estimate": None},
        serializer,
    )
    return directory


def write_live(directory, *, age=0.0, **batch):
    payload = {
        "schema": SCHEMA,
        "generated_at": time() - age,
        "source": "live",
        "batch": {
            "name": directory.name,
            "dir": str(directory),
            "elapsed_seconds": 30.0,
            "coverage": 0.8,
            "max_retries": 1,
            "experiments": 2,
            "stages": 1,
            "counts": {
                "running": 2,
                "pending": 0,
                "succeeded": 0,
                "failed": 0,
                "cancelled": 0,
            },
            "remaining": None,
            **batch,
        },
        "stages": [],
        "workers": [{"run": "abc123", "stage_index": 0}],
        "resources": {"devices": [], "host": {"total_kb": 64 * GIB}},
        "schedule": {"pending_runs": 0, "running_runs": 2},
        "events": [],
    }
    (directory / LIVE_FILE).write_text(json.dumps(payload))
    return payload


# --------------------------------------------------------------------------
# The snapshot the experiment process builds for itself.
# --------------------------------------------------------------------------


@POSIX
def test_live_gantt_uses_the_effective_clock_and_keeps_retry_history(
    tmp_path, monkeypatch
):
    batch = make_batch(tmp_path, count=1)
    scheduler = GpuScheduler(batch)
    run_id = batch.experiments[0].run_id
    live_worker(batch, run_id)
    worker = scheduler.workers[run_id]
    monkeypatch.setattr(
        "expman.observation.snapshot.perf_counter", lambda: worker.started + 30.0
    )
    # The session resumed at second 600. At effective second 700, the worker
    # has run for 30 seconds, so its start belongs at second 670.
    batch._elapsed_seconds = 700.0
    batch.state.session_marks = [600.0]
    previous = {
        "run_id": run_id,
        "stage": 0,
        "attempt": 1,
        "device": 0,
        "start_seconds": 580.0,
        "end_seconds": 600.0,
        "status": Status.CANCELLED.value,
        "reused": False,
    }
    batch.state.stage_history.append(previous)

    snapshot = build_snapshot(batch)

    rows = snapshot["gantt"]
    assert [(row["attempt"], row["status"]) for row in rows] == [
        (1, "cancelled"),
        (2, "running"),
    ]
    assert rows[1]["start_seconds"] == 670.0
    assert rows[1]["end_seconds"] == 700.0
    assert snapshot["batch"]["session_marks"] == [600.0]

    # The durable completion wins even if a publisher can still see the worker.
    batch.state.stage_history.append(
        {
            **previous,
            "attempt": 2,
            "start_seconds": 670.0,
            "end_seconds": 700.0,
            "status": Status.SUCCEEDED.value,
        }
    )
    completed = build_snapshot(batch)["gantt"]
    assert len(completed) == 2
    assert completed[-1]["status"] == "succeeded"
    scheduler.workers.clear()
    batch.state.active_gpu.clear()
    assert build_snapshot(batch)["gantt"] == completed


def test_the_archived_path_exposes_the_folds_it_recorded(tmp_path):
    directory = tmp_path / "batch"
    directory.mkdir()
    manifest = {
        "pipeline": [{"class": "pkg.Stage"}],
        "max_retries": 1,
        "experiments": [{"run_id": "a" * 32, "attempts": []}],
        "queue": [],
        "active_gpu": {},
        "stage_history": [
            {
                "run_id": "a" * 32,
                "stage": 0,
                "attempt": 1,
                "status": Status.SUCCEEDED.value,
                "reused": False,
                "start_seconds": 1.0,
                "end_seconds": 5.0,
            }
        ],
        "session_marks": [12.0, 4.0],
    }
    BatchStore(directory).save_manifest(manifest)

    snapshot = read_batch(directory, 0)

    assert snapshot["batch"]["session_marks"] == [4.0, 12.0]
    rows = snapshot["gantt"]
    assert len(rows) == 1
    assert rows[0]["start_seconds"] == 1.0
    assert rows[0]["end_seconds"] == 5.0


@POSIX
@pytest.mark.simulated_gpu
def test_finished_batch_publishes_the_snapshot_read_by_the_dashboard(tmp_path):
    batch = make_batch(tmp_path)
    batch.run()

    snapshot = read_batch(batch.output_dir, 3)

    assert snapshot["source"] == "last"
    assert snapshot["status"] == "finished"
    assert snapshot["running"] is False
    assert snapshot["schema"] == SCHEMA
    assert snapshot["batch"]["counts"]["succeeded"] == 2
    stage = snapshot["stages"][0]
    assert stage["name"] == "Work"
    assert stage["groups_done"] == 2
    assert stage["groups_remaining"] == 0
    assert stage["samples"]["measured"] == 2
    assert snapshot["resources"]["host"]["total_kb"] == pytest.approx(64 * GIB)
    assert snapshot["resources"]["host"]["utilization_percent"] == 31.5
    card = snapshot["resources"]["devices"][0]
    assert card["total_kb"] == pytest.approx(24 * GIB)
    assert card["free_kb"] == pytest.approx(16 * GIB)
    assert card["utilization_percent"] == 72
    assert snapshot["schedule"]["pending_runs"] == 0
    kinds = [event["kind"] for event in snapshot["events"]]
    assert "launch" in kinds
    assert "finish" in kinds
    spans = [(row["start_seconds"], row["end_seconds"]) for row in snapshot["gantt"]]
    assert {row["run"] for row in snapshot["gantt"]} == {
        run.run_id[:6] for run in batch.experiments
    }
    assert sorted(spans) == sorted(
        (attempt["start_seconds"], attempt["end_seconds"])
        for attempt in batch.state.stage_history
    )
    assert all(0 <= start <= end <= batch.elapsed_seconds + 1 for start, end in spans)


@POSIX
def test_stage_samples_match_the_observations_used_by_duration_estimates():
    history = [
        {"stage": 0, "status": status, "duration": duration}
        for status, duration in [
            ("succeeded", 2),
            ("succeeded", None),
            ("cancelled", 3),
            ("failed", 4),
        ]
    ]
    history.append({"stage": 0, "status": "succeeded", "reused": True})
    assert stage_samples(history)[0] == {"measured": 1, "reused": 1, "censored": 1}


def test_a_snapshot_never_fits_a_model(tmp_path):
    batch = seed_finished_first(make_batch(tmp_path))
    scheduler = GpuScheduler(batch)

    build_snapshot(batch)

    # The dashboard reads, the scheduler fits: an open page must not be able to
    # change what admission decides.
    assert scheduler.stages._duration_models == {}
    assert scheduler.stages._peak_models == {}


@POSIX
def test_gate_uses_current_plan_and_actual_gpu_headroom(tmp_path):
    batch = make_batch(tmp_path, count=1)
    scheduler = GpuScheduler(batch)
    memory = {0: DeviceMemory("GPU-test-0", 24 * 1024, 190)}
    host = Host().sample()
    batch.live.host_memory = {
        "available_kb": host.available_kb,
        "admits_next": True,
        "observed_at": time(),
    }
    assert not scheduler._launch_plan(memory, host)
    schedule = build_snapshot(batch)["schedule"]
    assert schedule["gate"] == "closed"
    assert schedule["plan"]["host_fits"] is True
    assert schedule["plan"]["gpu_fits"] is False
    assert schedule["plan"]["launched"] == []

    assert scheduler._launch_plan(Memory([0]).sample(), host)
    assert build_snapshot(batch)["schedule"]["gate"] == "open"
    batch.live.host_memory["observed_at"] = scheduler.last_plan["at"] + 1
    schedule = build_snapshot(batch)["schedule"]
    assert schedule["plan"] is None
    assert schedule["gate"] == "waiting"


@POSIX
def test_a_running_attempt_reports_allocation_against_actual_use(tmp_path):
    batch = seed_finished_first(make_batch(tmp_path))
    GpuScheduler(batch)
    run_id = batch.state.queue[0]
    live_worker(batch, run_id)

    snapshot = build_snapshot(batch)

    worker = snapshot["workers"][0]
    assert worker["run"] == run_id[:6]
    assert worker["attempt"] == 2
    assert worker["stage"] == "Work"
    assert worker["elapsed_seconds"] == pytest.approx(30.0, abs=2.0)
    assert worker["expected"]["center_seconds"] == 90.0
    assert worker["host"]["reserved_kb"] == 8 * GIB
    assert worker["host"]["resident_kb"] == 3 * GIB
    assert worker["gpu"]["budget_kb"] == 6 * GIB
    assert worker["gpu"]["used_kb"] == 2 * GIB
    assert worker["gpu"]["cap_kb"] == 7 * GIB

    card = snapshot["resources"]["devices"][0]
    assert card["expman_actual_kb"] == 2 * GIB
    assert card["expman_budget_kb"] == 6 * GIB
    assert card["other_kb"] == pytest.approx(6 * GIB)
    assert worker["gpu"]["peak_used_kb"] == 3 * GIB
    host = snapshot["resources"]["host"]
    assert host["expman_actual_kb"] == 3 * GIB
    assert host["expman_budget_kb"] == 8 * GIB
    assert host["other_kb"] == pytest.approx(21 * GIB)

    assert snapshot["schedule"]["running_runs"] == 1
    assert snapshot["schedule"]["running_by_stage"][0] == 1


@POSIX
@pytest.mark.parametrize(
    ("verified", "observed", "displayed"),
    [
        (False, 5 * GIB, None),
        (True, 5 * GIB, 5 * GIB),
        (True, None, 8 * GIB),
        (True, 0, None),
    ],
)
def test_host_caps_only_display_verified_enforcement(
    tmp_path, verified, observed, displayed
):
    batch = make_batch(tmp_path, count=1)
    scheduler = GpuScheduler(batch)
    run_id = batch.experiments[0].run_id
    live_worker(batch, run_id)
    worker = scheduler.workers[run_id]
    worker.limit = SimpleNamespace(cap_kb=8 * GIB, verified=verified, capped=verified)
    worker.host_limit_kb = observed
    worker.host_usage_source = "proc"
    worker.host_usage_error = "cgroup usage unavailable"

    host = build_snapshot(batch)["workers"][0]["host"]

    assert host["cap_kb"] == displayed
    assert host["requested_cap_kb"] == 8 * GIB
    assert host["capped"] is (displayed is not None)
    assert host["usage_source"] == "proc"
    assert host["usage_error"] == "cgroup usage unavailable"


@POSIX
def test_unavailable_worker_usage_does_not_become_zero_in_the_host_total(tmp_path):
    batch = make_batch(tmp_path, count=1)
    scheduler = GpuScheduler(batch)
    run_id = batch.experiments[0].run_id
    live_worker(batch, run_id)
    worker = scheduler.workers[run_id]
    worker.resident_kb = 0.0
    worker.peak_kb = 0.0
    worker.host_usage_source = "unavailable"

    snapshot = build_snapshot(batch)

    assert snapshot["workers"][0]["host"]["resident_kb"] is None
    assert snapshot["workers"][0]["host"]["peak_kb"] is None
    assert snapshot["resources"]["host"]["expman_actual_kb"] is None
    assert snapshot["resources"]["host"]["other_kb"] is None
    assert snapshot["resources"]["host"]["expman_budget_kb"] == 8 * GIB


@POSIX
def test_the_estimate_publishes_per_stage_group_counts_and_duration_bounds(tmp_path):
    batch = seed_finished_first(make_batch(tmp_path, count=3))
    scheduler = GpuScheduler(batch)

    remaining = estimate_parallel(scheduler, 0.8)

    view = batch.live.schedule_view
    assert remaining.lower_seconds is not None
    assert view["degraded"] is None
    assert view["groups"] == 2
    assert view["pending_runs"] == 2
    stage = view["stages"][0]
    assert stage["groups"] == 2
    assert stage["spans"] == 2
    assert stage["unit_lower_seconds"] > 0
    assert stage["unit_lower_seconds"] <= stage["unit_upper_seconds"]


@POSIX
def test_a_snapshot_without_a_scheduler_still_lists_the_stages(tmp_path):
    snapshot = build_snapshot(make_batch(tmp_path))

    assert snapshot["stages"][0]["name"] == "Work"
    assert snapshot["stages"][0]["known"] is False
    assert snapshot["stages"][0]["groups_remaining"] is None
    assert snapshot["workers"] == []


# --------------------------------------------------------------------------
# Publishing live state from the experiment process.
# --------------------------------------------------------------------------


def test_live_settings_and_environment_control_publishing(tmp_path, monkeypatch):
    monkeypatch.delenv("EXPMAN_LIVE", raising=False)
    assert live_enabled() is True
    monkeypatch.setattr(settings, "live_enabled", False)
    assert live_enabled() is False
    monkeypatch.setattr(settings, "live_interval", 2.5)
    assert LivePublisher(Broken(tmp_path)).interval == 2.5
    monkeypatch.setenv("EXPMAN_LIVE", "off")
    assert live_enabled() is False
    monkeypatch.setenv("EXPMAN_LIVE", "1")
    assert live_enabled() is True


def test_publishing_degrades_instead_of_raising(tmp_path):
    publisher = LivePublisher(Broken(tmp_path))

    publisher.publish()

    payload = json.loads((tmp_path / LIVE_FILE).read_text())
    assert payload["schema"] == SCHEMA
    assert payload["degraded"].startswith("AttributeError")
    assert list(tmp_path.glob("*.tmp")) == []


@POSIX
@pytest.mark.simulated_gpu
def test_publishing_is_skipped_when_switched_off(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPMAN_LIVE", "off")
    batch = make_batch(tmp_path, count=1)

    batch.run()

    assert not (batch.output_dir / LIVE_FILE).exists()


# --------------------------------------------------------------------------
# Reading a batch that is not running, or not publishing.
# --------------------------------------------------------------------------


def test_a_batch_without_a_snapshot_falls_back_to_its_records(tmp_path):
    directory = write_batch_dir(tmp_path, "run_a", runs=3)

    snapshot = read_batch(directory, 7)

    assert snapshot["source"] == "archive"
    assert snapshot["id"] == 7
    assert snapshot["status"] == "finished"
    assert snapshot["batch"]["counts"]["succeeded"] == 3
    assert snapshot["batch"]["elapsed_seconds"] == 120.0
    assert [stage["name"] for stage in snapshot["stages"]] == ["Stage0"]
    assert snapshot["stages"][0]["approximate"] is True
    assert snapshot["stages"][0]["groups_total"] == 3
    assert snapshot["stages"][0]["unit_lower_seconds"] is None
    assert snapshot["workers"] == []
    assert snapshot["resources"] is None
    assert snapshot["events"] == []


def test_a_live_stop_channel_plus_a_fresh_snapshot_reads_as_running(tmp_path):
    directory = write_batch_dir(tmp_path, "run_a")
    write_live(directory)
    with run_lock(directory), stop_on_sigterm(directory):
        snapshot = read_batch(directory, 0)

    assert snapshot["source"] == "live"
    assert snapshot["status"] == "running"
    assert snapshot["running"] is True
    assert snapshot["age_seconds"] < 3
    assert len(snapshot["workers"]) == 1


def test_a_stale_snapshot_reads_as_the_last_state_it_reached(tmp_path):
    directory = write_batch_dir(tmp_path, "run_a", status="cancelled")
    write_live(directory, age=60.0)

    snapshot = read_batch(directory, 0)

    assert snapshot["source"] == "last"
    assert snapshot["status"] == "stopped"
    assert snapshot["running"] is False
    assert snapshot["age_seconds"] > 30
    # The last published readings survive, so a stopped batch still explains
    # what it was doing when it stopped.
    assert snapshot["resources"]["host"]["total_kb"] == 64 * GIB


def test_the_terminal_status_reflects_the_attempts(tmp_path):
    failed = read_batch(write_batch_dir(tmp_path, "bad", status="failed"), 0)
    stopped = read_batch(write_batch_dir(tmp_path, "cut", status="cancelled"), 0)
    pending = read_batch(write_batch_dir(tmp_path, "wait", status="pending"), 0)

    assert failed["status"] == "failed"
    assert stopped["status"] == "stopped"
    assert pending["status"] == "stopped"
    assert pending["batch"]["counts"]["pending"] == 2


def test_stale_snapshot_cannot_hide_a_later_run_completion(tmp_path):
    directory = write_batch_dir(tmp_path, "done", status="succeeded")
    write_live(directory, age=60)
    manifest_time = (directory / "batch.pkl").stat().st_mtime_ns
    os.utime(directory / LIVE_FILE, ns=(manifest_time - 1, manifest_time - 1))
    snapshot = read_batch(directory, 0)
    assert snapshot["source"] == "archive"
    assert snapshot["status"] == "finished"


def test_a_directory_without_records_is_reported_not_raised(tmp_path):
    directory = tmp_path / "empty"
    directory.mkdir()

    snapshot = read_batch(directory, 0)

    assert snapshot["source"] == "missing"
    assert snapshot["id"] == 0
    assert snapshot["stages"] == []
    assert snapshot["resources"] is None


def test_only_directories_holding_records_become_batches(tmp_path):
    root = tmp_path / "batches"
    assert ensure_ids(root) == {}
    assert not root.exists()
    root.mkdir()
    assert ensure_ids(root) == {}
    assert not list(root.iterdir())

    write_batch_dir(root, "run_a")
    write_batch_dir(root, "run_b")
    (root / "notes").mkdir()

    ids = ensure_ids(root)

    assert sorted(ids) == [0, 1]
    assert {directory.name for directory in ids.values()} == {"run_a", "run_b"}


def test_concurrent_discovery_keeps_the_id_each_caller_observed(tmp_path):
    workers = 8
    start = Barrier(workers)

    def register(index):
        start.wait(timeout=10)
        directory = write_batch_dir(tmp_path, f"run-{index}")
        ids = ensure_ids(tmp_path)
        return next(batch_id for batch_id, path in ids.items() if path == directory)

    with ThreadPoolExecutor(workers) as pool:
        assigned = list(pool.map(register, range(workers)))

    expected = {batch_id: tmp_path / f"run-{i}" for i, batch_id in enumerate(assigned)}
    assert sorted(expected) == list(range(workers))
    assert ensure_ids(tmp_path) == expected
    dropped = assigned[0]
    (expected.pop(dropped) / "batch.pkl").unlink()
    write_batch_dir(tmp_path, "later")
    assert ensure_ids(tmp_path) == {**expected, workers: tmp_path / "later"}


# --------------------------------------------------------------------------
# The dashboard process.
# --------------------------------------------------------------------------


class TestDashboard:
    def get(self, request):
        """Fetch a URL or a prepared request, treating an error as a response."""
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, response.read(), response.headers
        except HTTPError as error:
            return error.code, error.read(), error.headers

    @pytest.fixture
    def runs(self, tmp_path):
        """Give run_a the earlier mtime so the first discovery assigns it ID 0."""
        root = tmp_path / "batches"
        write_batch_dir(root, "run_a", runs=3)
        write_batch_dir(root, "run_b", runs=1, status="failed")
        now = time()
        os.utime(root / "run_a", (now - 10, now - 10))
        os.utime(root / "run_b", (now, now))
        return root

    def test_the_pages_assets_and_health_endpoint_are_served(self, runs):
        dashboard = serve(runs, port=0, interval=1.0)
        try:
            status, body, headers = self.get(dashboard.url)
            assert status == 200
            assert b"expman" in body
            assert headers["Content-Type"].startswith("text/html")

            status, body, _headers = self.get(dashboard.url + "b/0/")
            assert status == 200
            assert b"/static/results.js" in body
            assert self.get(dashboard.url + "b/0/results/")[:2] == (200, body)
            for asset in ("results.js", "results.css"):
                assert self.get(dashboard.url + "static/" + asset)[0] == 200

            status, _body, headers = self.get(dashboard.url + "static/app.js")
            assert status == 200
            assert headers["Content-Type"].startswith("text/javascript")
            status, body, _headers = self.get(dashboard.url + "api/health")
            assert status == 200
            assert json.loads(body) == {"status": "ok", "schema": SCHEMA}

            assert self.get(dashboard.url + "static/nope.js")[0] == 404
            assert self.get(dashboard.url + "api/nope")[0] == 404
            assert self.get(dashboard.url + "b/nope/")[0] == 404
        finally:
            dashboard.close()

    def test_overview_ids_open_the_corresponding_batch_detail(self, runs):
        dashboard = serve(runs, port=0, interval=1.0)
        try:
            dashboard.refresh()
            status, body, _headers = self.get(dashboard.url + "api/batches")
            assert status == 200
            payload = json.loads(body)
            assert payload["schema"] == SCHEMA
            finished, failed = payload["batches"]
            assert (finished["name"], failed["name"]) == ("run_a", "run_b")
            assert finished["counts"]["succeeded"] == finished["experiments"] == 3
            assert (finished["stages"], finished["workers"]) == (1, 0)
            assert failed["status"] == "failed"
            status, body, _headers = self.get(
                dashboard.url + f"api/batch/{finished['id']}"
            )
            assert status == 200
            detail = json.loads(body)
            assert (detail["id"], detail["batch"]["name"]) == (finished["id"], "run_a")
            assert self.get(dashboard.url + "api/batch/99")[0] == 404
        finally:
            dashboard.close()

    def test_result_layout_only_recomputes_for_changed_measurements(
        self, tmp_path, monkeypatch
    ):
        directory = write_batch_dir(tmp_path, "example", runs=2, status="pending")
        store = BatchStore(directory)
        manifest = store.read_manifest()
        first, second = (entry["run_id"] for entry in manifest["experiments"])
        for x, run_id in enumerate((first, second), start=1):
            write_record(
                directory / "experiments" / run_id / "config.pkl",
                {"x": x},
                PickleSerializer(),
            )
        metrics = MetricStore(directory / "metrics.sqlite3")
        layout = Mock(wraps=layout_results)
        monkeypatch.setattr("expman.webui.server.layout_results", layout)
        dashboard = serve(tmp_path, port=0)
        url = dashboard.url + "api/batch/0/results"

        def read_rows(request=url):
            status, content, _ = self.get(request)
            assert status == 200
            table = json.loads(content)
            assert table["name"] == directory.name
            labels = {column["id"]: column["label"] for column in table["columns"]}
            rows = [
                {labels[key]: value for key, value in row.items()}
                for row in table["rows"]
            ]
            return sorted(rows, key=lambda row: row["x"])

        try:
            assert self.get(dashboard.url + "api/batch/99/results")[0] == 404
            empty = [{"x": 1}, {"x": 2}]
            assert read_rows() == empty
            initial_calls = layout.call_count
            assert initial_calls <= 1
            assert read_rows() == empty
            assert layout.call_count == initial_calls

            manifest["queue"].remove(first)
            manifest["experiments"][0]["attempts"][0]["status"] = "running"
            store.save_manifest(manifest)
            dashboard.refresh()
            assert read_rows() == empty
            assert layout.call_count == initial_calls

            metrics.log(first, (0,), None, 1, [('["score"]', 5.0)])
            # Subsequent writes commit while the dashboard's read connection stays open.
            with closing(sqlite3.connect(metrics.path)) as writer:
                writer.execute("PRAGMA journal_mode=WAL")
            partial = [{"x": 1, "score": 5}, {"x": 2}]
            assert read_rows() == partial
            assert layout.call_count == initial_calls + 1
            assert read_rows() == partial
            assert layout.call_count == initial_calls + 1

            metrics.log(second, (0,), None, 1, [('["score"]', 5.0)])
            complete = [{"x": 1, "score": 5}, {"x": 2, "score": 5}]
            assert read_rows() == complete
            assert layout.call_count == initial_calls + 2

            status, _, headers = self.get(url)
            assert status == 200 and headers["ETag"]
            conditional = Request(url, headers={"If-None-Match": headers["ETag"]})
            metrics.log(second, (0,), None, 2, [('["score"]', 5.0)])
            dashboard.refresh()
            status, content, _ = self.get(conditional)
            assert (status, content) == (304, b"")
            assert layout.call_count == initial_calls + 2

            metrics.log(second, (0,), None, 2, [('["score"]', 8.0)])
            assert read_rows(conditional) == [
                {"x": 1, "score": 5},
                {"x": 2, "score": 8},
            ]
            assert layout.call_count == initial_calls + 3

            metrics.log(first, (0,), None, 1, [('["loss"]', 0.5)])
            assert read_rows() == [
                {"x": 1, "score": 5, "loss": 0.5},
                {"x": 2, "score": 8},
            ]
            assert layout.call_count == initial_calls + 4
        finally:
            dashboard.close()

    def test_serve_reads_the_configured_root_and_web_defaults_at_startup(
        self, runs, monkeypatch
    ):
        monkeypatch.setattr(settings, "batches_dir", runs)
        monkeypatch.setattr(settings, "web_port", 0)
        monkeypatch.setattr(settings, "web_interval", 12.5)
        dashboard = serve()
        try:
            assert dashboard._server.stream.interval == 12.5
            monkeypatch.setattr(settings, "batches_dir", runs / "different")
            dashboard.refresh()
            _, body, _ = self.get(dashboard.url + "api/batches")
            assert [entry["name"] for entry in json.loads(body)["batches"]] == [
                "run_a",
                "run_b",
            ]
        finally:
            dashboard.close()

    def test_the_stream_pushes_a_frame_for_the_watched_batch(self, runs):
        dashboard = serve(runs, port=0, interval=1.0)
        try:
            dashboard.refresh()
            with urlopen(dashboard.url + "api/stream?batch=0", timeout=5) as response:
                assert response.readline().startswith(b"retry:")
                while True:
                    line = response.readline()
                    if line.startswith(b"data:"):
                        payload = json.loads(line[len(b"data:") :])
                        break
            assert payload["schema"] == SCHEMA
            assert payload["batch"]["id"] == 0
            assert len(payload["batches"]) == 2
        finally:
            dashboard.close()

    def test_a_replaced_subscriber_does_not_unwatch_its_replacement(self, runs):
        # A page that reloads unsubscribes as it goes. That unsubscribe used
        # to drop the batch the connection after it had just asked for, so
        # the new page waited a heartbeat without a frame and rendered as a
        # batch that does not exist.
        stream = Stream(Workspace(runs), interval=1.0)
        stream.watch(0)
        stream.watch(0)
        stream.unwatch(0)
        stream.refresh()
        assert json.loads(stream.frame(0))["batch"]["id"] == 0
        stream.unwatch(0)
        stream.refresh()
        assert "batch" not in json.loads(stream.frame(0))

    def test_a_tick_that_cannot_read_a_batch_keeps_its_last_frame(self, runs):
        # The previous detail remains available when one refresh skips this batch.
        stream = Stream(Workspace(runs), interval=1.0)
        stream.watch(0)
        stream.refresh()
        assert json.loads(stream.frame(0))["batch"]["id"] == 0
        with patch.object(Workspace, "collect", return_value=([], {})):
            stream.refresh()
        assert json.loads(stream.frame(0))["batch"]["id"] == 0
        (runs / "run_a").rename(runs.parent / "removed")
        stream.refresh()
        assert json.loads(stream.frame(0))["batch"] is None

    def test_a_failed_build_does_not_empty_the_open_pages(self, runs):
        stream = Stream(Workspace(runs), interval=1.0)
        stream.watch(0)
        stream.refresh()
        with patch.object(Workspace, "collect", side_effect=RuntimeError("boom")):
            stream.refresh()
        assert json.loads(stream.frame(0))["batch"]["id"] == 0
        assert "boom" in json.loads(stream.frame(""))["degraded"]

    def test_stopping_only_signals_the_matching_running_batch(self, runs):
        endpoint = runs / "run_a" / STOP_FIFO
        os.mkfifo(endpoint)
        # Restore the intended discovery order before the first registry scan.
        now = time()
        os.utime(runs / "run_a", (now - 10, now - 10))
        dashboard = serve(runs, port=0)
        try:
            with os.fdopen(
                os.open(endpoint, os.O_RDONLY | os.O_NONBLOCK), "rb"
            ) as reader:
                stale = Request(
                    dashboard.url + "api/stop/0?name=removed", method="POST"
                )
                assert self.get(stale)[0] == 409
                status, body, _headers = self.get(
                    Request(dashboard.url + "api/stop/0?name=run_a", method="POST")
                )
                assert status == 200
                assert json.loads(body)["status"] == "stopping"
                assert reader.read(1) == b"x"
            status, body, _headers = self.get(
                Request(dashboard.url + "api/stop/0", method="POST")
            )
            assert status == 200
            assert json.loads(body)["status"] == "refused"
            assert (
                self.get(Request(dashboard.url + "api/stop/99", method="POST"))[0]
                == 404
            )
        finally:
            dashboard.close()

    def test_a_taken_port_falls_back_to_the_next_one(self, runs):
        with socket.socket() as blocker:
            blocker.bind(("127.0.0.1", 0))
            blocker.listen(1)
            taken = blocker.getsockname()[1]
            dashboard = serve(runs, port=taken, interval=1.0)
            try:
                assert taken < dashboard.port < taken + 10
                assert self.get(dashboard.url)[0] == 200
            finally:
                dashboard.close()

    def test_a_missing_tree_is_an_empty_list_not_an_error(self, tmp_path):
        dashboard = serve(tmp_path / "nowhere", port=0, interval=1.0)
        try:
            dashboard.refresh()
            status, body, _headers = self.get(dashboard.url + "api/batches")
            assert status == 200
            assert json.loads(body)["batches"] == []
        finally:
            dashboard.close()

    def test_a_busy_last_port_reports_only_valid_port_numbers(self, runs):
        with (
            patch(
                "expman.webui.server.DashboardServer", side_effect=OSError("busy")
            ) as server,
            pytest.raises(OSError, match=r"65535\.\.65535"),
        ):
            serve(runs, port=65535)
        assert server.call_count == 1

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"port": -1},
            {"port": 70000},
            {"port": True},
            {"host": ""},
            {"interval": 0},
            {"interval": float("inf")},
            {"interval": True},
        ],
    )
    def test_invalid_settings_are_refused(self, tmp_path, kwargs):
        with pytest.raises(ValueError):
            serve(tmp_path, **kwargs)


# --------------------------------------------------------------------------
# The command line.
# --------------------------------------------------------------------------


def test_web_command_passes_project_defaults_and_cli_overrides(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[tool.expman]\nweb_port = 9301\n")
    with patch("expman.cli.serve") as serve_dashboard:
        assert main(["web"]) == 0
        serve_dashboard.assert_called_with(Path("batches"), host="127.0.0.1", port=9301)
        assert main(["web", "--port", "0", "--batches-dir", "results"]) == 0
        serve_dashboard.assert_called_with(Path("results"), host="127.0.0.1", port=0)


@POSIX
def test_nested_progress_flows_from_worker_ipc_to_snapshot(tmp_path):
    class Child(Stage):
        def process(self, data, ctx):
            ctx.report_progress(1, total=5)
            ctx.report_progress(2, total=5)

    config = tmp_path / "nested.yaml"
    config.write_text("device: [0]\nseed: 0\nitem: 0\n")
    batch = batch_helpers.make_batch(
        Pipeline([Work, Work]), config, output_dir=tmp_path / "batch"
    )
    scheduler = GpuScheduler(batch)
    run_id = batch.experiments[0].run_id
    live_worker(batch, run_id)
    worker = scheduler.workers[run_id]
    worker.stage_index = 1
    context = RunContext(run_id=run_id, stage_id=1, attempt=worker.stage_attempt)
    Pipeline([Child]).run(ctx=context)
    emitted = [
        event for event in context.recorder.events if isinstance(event, ProgressEvent)
    ]
    assert [event.stage_id for event in emitted] == [0, 0]
    worker.channel = Mock()
    worker.channel.drain.return_value = [
        {"type": "event", "event": event} for event in context.recorder.events
    ]

    worker.read_messages(batch.recorder)

    assert worker.progress is emitted[-1]
    assert any(event is emitted[-1] for event in batch.recorder.events)
    current = build_snapshot(batch)["workers"][0]
    assert current["stage_index"] == 1
    assert current["progress"] == {
        "completed": 2,
        "total": 5,
        "unit": "step",
        "stage_id": 0,
    }

    # Even a new worker whose top-level ID matches the old child ID starts empty.
    live_worker(batch, run_id)
    fresh = scheduler.workers[run_id]
    assert fresh.progress is None
    assert build_snapshot(batch)["workers"][0]["progress"] is None
    # A retry must neither inherit the cache nor accept a stale-attempt IPC event.
    fresh.stage_index = 1
    fresh.stage_attempt = 3
    fresh.channel = Mock()
    fresh.channel.drain.return_value = [{"type": "event", "event": emitted[-1]}]
    fresh.read_messages(batch.recorder)
    assert fresh.progress is None
    assert build_snapshot(batch)["workers"][0]["progress"] is None
