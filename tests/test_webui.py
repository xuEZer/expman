import json
import os
import socket
from time import perf_counter, sleep, time
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
import yaml

from expman import Batch, Pipeline, Stage, Status, devices
from expman._duration_model import StageDuration
from expman.analysis import ANALYSIS_SCHEMA
from expman.cli import _parser
from expman.devices import DeviceMemory, HostMemory
from expman.parallel_estimation import estimate_parallel
from expman.registry import ensure_ids
from expman.scheduling import GpuScheduler, Worker
from expman.storage import PickleSerializer, write_record
from expman.webui import SCHEMA, build_snapshot, read_batch, serve, summarize
from expman.webui.live import LIVE_FILE, LivePublisher, live_enabled
from expman.webui.server import Stream, Workspace

POSIX = pytest.mark.skipif(os.name != "posix", reason="POSIX worker process groups")
GIB = 1024 * 1024


class Work(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {"markers": True, "item": True, "delay": True}

    def process(self, data, ctx):
        sleep(ctx.cfg["delay"])
        return ctx.cfg["item"]


class Memory:
    """A 24 GiB card with 16 GiB free, reported in the units nvidia-smi uses."""

    def __init__(self, devices):
        self.devices = devices

    def sample(self):
        return {
            device: DeviceMemory(f"GPU-test-{device}", 24 * 1024, 16 * 1024)
            for device in self.devices
        }


class Host:
    def sample(self):
        return HostMemory(64 * GIB, 40 * GIB, 0, 0)


class Broken:
    """A stand-in for a Batch whose attributes cannot be read at all."""

    def __init__(self, output_dir):
        self.output_dir = output_dir


@pytest.fixture(autouse=True)
def fake_devices(monkeypatch):
    monkeypatch.setattr(devices, "NvidiaMemory", Memory)
    monkeypatch.setattr(devices, "MeminfoMonitor", Host)
    monkeypatch.setattr(devices, "POLL_INTERVAL", 0.01)


def make_batch(root, *, count=2, delay=0.0):
    cfg = {"device": [0], "seed": 0, "markers": str(root), "delay": delay, "item": 0}
    path = root / "experiments.yaml"
    path.write_text(yaml.safe_dump(cfg) + f"grid:\n  item: {list(range(count))}\n")
    return Batch(Pipeline([Work]), path, output_dir=root / "batch")


def seed_finished_first(batch, *, duration=2.0):
    """Finish the first configuration so the rest has measured samples.

    The remaining configurations stay queued, which is the state a dashboard is
    normally opened in: work left to do, with a measured Stage behind it.
    """
    experiment = batch.experiments[0]
    for stage_index in range(len(experiment.pipeline.stages)):
        batch._stage_history.append(
            {
                "run_id": experiment.run_id,
                "stage": stage_index,
                "attempt": 1,
                "status": Status.SUCCEEDED.value,
                "stage_duration_seconds": duration,
                "peak_kb": 4096.0,
                "gpu_peak_kb": 1024.0,
                "capped": False,
                "gpu_capped": False,
                "reused": False,
            }
        )
    batch._stage_progress[experiment.run_id] = len(experiment.pipeline.stages)
    batch._queue.remove(experiment.run_id)
    return batch


def live_worker(batch, run_id, *, device=0, elapsed=30.0):
    """Install one running attempt with the readings a scheduler tick would leave."""
    experiment = next(item for item in batch.experiments if item.run_id == run_id)
    batch._scheduler.workers[run_id] = Worker(
        process=Mock(),
        experiment=experiment,
        device=device,
        uuid=f"GPU-test-{device}",
        root=batch.output_dir,
        started=perf_counter() - elapsed,
        stage_index=0,
        stage_attempt=2,
        host_reserve_kb=8 * GIB,
        gpu_reserve_kb=6 * GIB,
        gpu_cap_kb=7 * GIB,
        expected=StageDuration(center=90.0, lower=60.0, upper=150.0),
        resident_kb=3 * GIB,
        peak_kb=4 * GIB,
        gpu_allocated_kb=1 * GIB,
        gpu_reserved_kb=2 * GIB,
        gpu_peak_allocated_kb=1 * GIB,
        gpu_peak_reserved_kb=3 * GIB,
    )
    batch._active_gpu[run_id] = device
    batch._gpu_running_info[run_id] = {
        "device": device,
        "uuid": f"GPU-test-{device}",
        "concurrency": 1.0,
        "duration": elapsed,
        "expected_seconds": 90.0,
        "expected_lower_seconds": 60.0,
        "expected_upper_seconds": 150.0,
        "stage_index": 0,
        "resident_kb": 3 * GIB,
        "peak_kb": 4 * GIB,
        "gpu_allocated_kb": 1 * GIB,
        "gpu_reserved_kb": 2 * GIB,
        "gpu_peak_allocated_kb": 1 * GIB,
        "gpu_peak_reserved_kb": 3 * GIB,
    }
    batch._device_memory = {
        device: {
            "uuid": f"GPU-test-{device}",
            "total_kb": 24 * GIB,
            "free_kb": 16 * GIB,
            "observed_at": time(),
        }
    }
    batch._host_memory = {
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


def write_batch_dir(root, name, *, runs=2, status="succeeded", stages=1, pid=None):
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
        "version": 1,
        "pipeline": [
            {"class": f"demo.Stage{index}", "source": None} for index in range(stages)
        ],
        "pipeline_spec": None,
        "max_retries": 1,
        "experiments": experiments,
        "queue": [entry["run_id"] for entry in experiments] if pending else [],
        "active_gpu": {},
        "stage_progress": {},
        "stage_attempts": {},
        "stage_elapsed": {},
        "stage_history": [],
        "gpu_history": [],
        "estimation": {"version": 1, "execution": "gpu", "coverage": 0.8},
    }
    serializer = PickleSerializer()
    write_record(directory / "batch.pkl", manifest, serializer)
    write_record(
        directory / "timing.pkl",
        {"version": 1, "elapsed_seconds": 120.0, "estimate": None},
        serializer,
    )
    if pid is not None:
        (directory / "expman.pid").write_text(str(pid))
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
def test_a_finished_batch_reports_stage_and_resource_facts(tmp_path):
    batch = make_batch(tmp_path)
    batch.run(progress=False)

    snapshot = build_snapshot(batch)

    assert snapshot["schema"] == SCHEMA
    assert "degraded" not in snapshot
    assert snapshot["batch"]["experiments"] == 2
    assert snapshot["batch"]["counts"]["succeeded"] == 2
    assert [stage["name"] for stage in snapshot["stages"]] == ["Work"]
    stage = snapshot["stages"][0]
    assert stage["known"] is True
    assert stage["groups_done"] == 2
    assert stage["groups_remaining"] == 0
    assert stage["samples"]["measured"] == 2
    assert snapshot["resources"]["host"]["total_kb"] == pytest.approx(64 * GIB)
    card = snapshot["resources"]["devices"][0]
    assert card["total_kb"] == pytest.approx(24 * GIB)
    assert card["free_kb"] == pytest.approx(16 * GIB)
    assert snapshot["schedule"]["pending_runs"] == 0
    kinds = [event["kind"] for event in snapshot["events"]]
    assert "launch" in kinds
    assert "finish" in kinds


@POSIX
def test_a_snapshot_never_fits_a_model(tmp_path):
    batch = seed_finished_first(make_batch(tmp_path))
    scheduler = GpuScheduler(batch)

    build_snapshot(batch)

    # The dashboard reads, the scheduler fits: an open page must not be able to
    # change what admission decides.
    assert scheduler._stage_duration_models == {}
    assert scheduler._stage_peak_models == {}


@POSIX
def test_a_running_attempt_reports_allocation_against_actual_use(tmp_path):
    batch = seed_finished_first(make_batch(tmp_path))
    GpuScheduler(batch)
    run_id = batch._queue[0]
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
    assert worker["gpu"]["reserved_kb"] == 2 * GIB
    assert worker["gpu"]["cap_kb"] == 7 * GIB

    card = snapshot["resources"]["devices"][0]
    assert card["expman_actual_kb"] == 2 * GIB
    assert card["expman_allocated_kb"] == 1 * GIB
    assert card["expman_budget_kb"] == 6 * GIB
    assert card["other_kb"] == pytest.approx(6 * GIB)
    host = snapshot["resources"]["host"]
    assert host["expman_actual_kb"] == 3 * GIB
    assert host["expman_budget_kb"] == 8 * GIB
    assert host["other_kb"] == pytest.approx(21 * GIB)

    assert snapshot["schedule"]["running_runs"] == 1
    assert snapshot["schedule"]["running_by_stage"][0] == 1


@POSIX
def test_the_estimate_publishes_the_unit_interval_the_page_multiplies(tmp_path):
    batch = seed_finished_first(make_batch(tmp_path, count=3))
    scheduler = GpuScheduler(batch)

    remaining = estimate_parallel(scheduler, 0.8)

    view = batch._schedule_view
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
def test_a_finished_batch_keeps_its_published_unit_intervals(tmp_path):
    batch = seed_finished_first(make_batch(tmp_path, count=3))
    scheduler = GpuScheduler(batch)
    estimate_parallel(scheduler, 0.8)
    batch._queue.clear()

    finished = estimate_parallel(scheduler, 0.8)

    assert finished.lower_seconds == 0.0
    assert batch._schedule_view["groups"] == 0
    assert batch._schedule_view["stages"][0]["spans"] == 2


@POSIX
def test_a_snapshot_without_a_scheduler_still_lists_the_stages(tmp_path):
    snapshot = build_snapshot(make_batch(tmp_path))

    assert snapshot["stages"][0]["name"] == "Work"
    assert snapshot["stages"][0]["known"] is False
    assert snapshot["stages"][0]["groups_remaining"] is None
    assert snapshot["workers"] == []


def test_a_snapshot_degrades_instead_of_raising(tmp_path):
    snapshot = build_snapshot(Broken(tmp_path))

    assert snapshot["schema"] == SCHEMA
    assert snapshot["degraded"].startswith("AttributeError")


# --------------------------------------------------------------------------
# Publishing live state from the experiment process.
# --------------------------------------------------------------------------


def test_publishing_is_on_by_default_and_can_be_switched_off(monkeypatch):
    monkeypatch.delenv("EXPMAN_LIVE", raising=False)
    assert live_enabled() is True
    monkeypatch.setenv("EXPMAN_LIVE", "off")
    assert live_enabled() is False
    monkeypatch.setenv("EXPMAN_LIVE", "1")
    assert live_enabled() is True


def test_publishing_degrades_instead_of_raising(tmp_path):
    publisher = LivePublisher(Broken(tmp_path))

    publisher.publish()

    payload = json.loads((tmp_path / LIVE_FILE).read_text())
    assert payload["degraded"].startswith("AttributeError")
    assert list(tmp_path.glob("*.tmp")) == []


@POSIX
def test_a_batch_publishes_live_state_while_it_runs(tmp_path):
    batch = make_batch(tmp_path)

    batch.run(progress=False)

    payload = json.loads((batch.output_dir / LIVE_FILE).read_text())
    assert payload["schema"] == SCHEMA
    assert payload["batch"]["counts"]["succeeded"] == 2
    assert payload["resources"]["host"]["total_kb"] == pytest.approx(64 * GIB)


@POSIX
def test_publishing_is_skipped_when_switched_off(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPMAN_LIVE", "off")
    batch = make_batch(tmp_path)

    batch.run(progress=False)

    assert not (batch.output_dir / LIVE_FILE).exists()


@POSIX
def test_a_finished_batch_reads_back_from_the_snapshot_it_left(tmp_path):
    batch = make_batch(tmp_path)
    batch.run(progress=False)

    snapshot = read_batch(batch.output_dir, 3)

    assert snapshot["source"] == "last"
    assert snapshot["status"] == "finished"
    assert snapshot["running"] is False
    assert snapshot["id"] == 3
    assert snapshot["batch"]["counts"]["succeeded"] == 2
    assert snapshot["batch"]["elapsed_seconds"] > 0
    assert [stage["name"] for stage in snapshot["stages"]] == ["Work"]


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


def test_a_live_pid_plus_a_fresh_snapshot_reads_as_running(tmp_path):
    directory = write_batch_dir(tmp_path, "run_a")
    write_live(directory)
    (directory / "expman.pid").write_text(str(os.getpid()))

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


def test_a_directory_without_records_is_reported_not_raised(tmp_path):
    directory = tmp_path / "empty"
    directory.mkdir()

    snapshot = read_batch(directory, 0)

    assert snapshot["source"] == "missing"
    assert snapshot["id"] == 0
    assert snapshot["stages"] == []
    assert snapshot["resources"] is None


def test_the_list_entry_carries_what_the_table_renders(tmp_path):
    directory = write_batch_dir(tmp_path, "run_a", runs=3)

    entry = summarize(read_batch(directory, 4))

    assert entry["id"] == 4
    assert entry["name"] == "run_a"
    assert entry["status"] == "finished"
    assert entry["counts"]["succeeded"] == 3
    assert entry["experiments"] == 3
    assert entry["stages"] == 1
    assert entry["workers"] == 0


def test_only_directories_holding_records_become_batches(tmp_path):
    write_batch_dir(tmp_path / "runs", "run_a")
    write_batch_dir(tmp_path / "runs", "run_b")
    (tmp_path / "runs" / "notes").mkdir()

    ids = ensure_ids(tmp_path / "runs")

    assert sorted(ids) == [0, 1]
    assert {directory.name for directory in ids.values()} == {"run_a", "run_b"}


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
        """Two batch directories with a fixed age order, so IDs are stable.

        ``scan_runs`` assigns IDs oldest-first, and two directories written in
        the same instant would otherwise swap IDs between runs.
        """
        root = tmp_path / "runs"
        write_batch_dir(root, "run_a", runs=3)
        write_batch_dir(root, "run_b", runs=1, status="failed")
        now = time()
        os.utime(root / "run_a", (now - 10, now - 10))
        os.utime(root / "run_b", (now, now))
        return root

    def test_the_pages_and_their_assets_are_served(self, runs):
        dashboard = serve(runs, port=0, interval=1.0)
        try:
            dashboard.refresh()
            status, body, headers = self.get(dashboard.url)
            assert status == 200
            assert b"expman" in body
            assert headers["Content-Type"].startswith("text/html")

            status, body, _headers = self.get(dashboard.url + "b/0/")
            assert status == 200
            assert b"app.js" in body

            status, _body, headers = self.get(dashboard.url + "static/app.js")
            assert status == 200
            assert headers["Content-Type"].startswith("text/javascript")

            assert self.get(dashboard.url + "static/nope.js")[0] == 404
            assert self.get(dashboard.url + "api/nope")[0] == 404
            assert self.get(dashboard.url + "b/nope/")[0] == 404
        finally:
            dashboard.close()

    def test_the_list_endpoint_lists_every_batch(self, runs):
        dashboard = serve(runs, port=0, interval=1.0)
        try:
            dashboard.refresh()
            status, body, _headers = self.get(dashboard.url + "api/batches")
            assert status == 200
            payload = json.loads(body)
            assert payload["schema"] == SCHEMA
            assert [entry["name"] for entry in payload["batches"]] == ["run_a", "run_b"]
            assert payload["batches"][1]["status"] == "failed"
        finally:
            dashboard.close()

    def test_the_batch_endpoint_returns_one_snapshot(self, runs):
        dashboard = serve(runs, port=0, interval=1.0)
        try:
            status, body, _headers = self.get(dashboard.url + "api/batch/0")
            assert status == 200
            payload = json.loads(body)
            assert payload["id"] == 0
            assert payload["batch"]["name"] == "run_a"
            assert self.get(dashboard.url + "api/batch/99")[0] == 404
        finally:
            dashboard.close()

    def test_the_analysis_endpoint_reports_partial_results(self, runs):
        from expman.metrics import MetricStore

        directory = runs / "run_a"
        run_id = "0" * 32
        config_dir = directory / "experiments" / run_id
        config_dir.mkdir(parents=True)
        write_record(config_dir / "config.pkl", {"lr": 1}, PickleSerializer())
        MetricStore(directory / "metrics.sqlite3").log(
            run_id, (0,), -1, 1, [(json.dumps(["score"]), 1.0)]
        )
        # Writing into run_a refreshed its mtime past run_b's; keep run_a the
        # oldest so it still owns ID 0.
        now = time()
        os.utime(directory, (now - 10, now - 10))
        dashboard = serve(runs, port=0, interval=1.0)
        try:
            status, body, _headers = self.get(dashboard.url + "api/analysis/0")
            assert status == 200
            payload = json.loads(body)
            assert payload["schema"] == ANALYSIS_SCHEMA
            coverage = payload["coverage"]
            # The other two runs of run_a have no config.pkl on disk: they
            # are skipped with a reason, not fatal.
            assert coverage["runs_total"] == 3
            assert coverage["runs_analyzed"] == 1
            assert [item["run_id"] for item in coverage["skipped"]] == [
                f"{index:032x}" for index in (1, 2)
            ]
            assert {item["name"] for item in payload["metrics"]} == {
                "score",
                "run_duration_seconds",
            }
            assert self.get(dashboard.url + "api/analysis/99")[0] == 404
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
        assert stream.watched == {0: 1}
        stream.unwatch(0)
        assert stream.watched == {}

    def test_a_tick_that_cannot_read_a_batch_keeps_its_last_frame(self, runs):
        # One unreadable tick is one stale second, not a missing batch.
        stream = Stream(Workspace(runs), interval=1.0)
        stream.watch(0)
        stream.refresh()
        assert json.loads(stream.frame(0))["batch"]["id"] == 0
        with patch.object(Workspace, "collect", return_value=([], {})):
            stream.refresh()
        assert json.loads(stream.frame(0))["batch"]["id"] == 0

    def test_a_failed_build_does_not_empty_the_open_pages(self, runs):
        stream = Stream(Workspace(runs), interval=1.0)
        stream.watch(0)
        stream.refresh()
        with patch.object(Workspace, "collect", side_effect=RuntimeError("boom")):
            stream.refresh()
        assert json.loads(stream.frame(0))["batch"]["id"] == 0
        assert "boom" in json.loads(stream.frame(""))["degraded"]

    def test_a_read_only_dashboard_refuses_to_stop(self, runs):
        dashboard = serve(runs, port=0, interval=1.0, allow_stop=False)
        try:
            status, _body, _headers = self.get(
                Request(dashboard.url + "api/stop/0", method="POST")
            )
            assert status == 403
        finally:
            dashboard.close()

    def test_stopping_signals_the_recorded_pid(self, runs, monkeypatch):
        signals = []
        monkeypatch.setattr(
            os, "kill", lambda pid, sig: signals.append((pid, sig)) if sig else None
        )
        (runs / "run_a" / "expman.pid").write_text("424242")
        # Writing into run_a refreshed its mtime past run_b's; keep run_a the
        # oldest so it still owns ID 0.
        now = time()
        os.utime(runs / "run_a", (now - 10, now - 10))
        dashboard = serve(runs, port=0, interval=1.0, allow_stop=True)
        try:
            status, body, _headers = self.get(
                Request(dashboard.url + "api/stop/0", method="POST")
            )
            assert status == 200
            assert json.loads(body)["status"] == "stopping"
            assert signals == [(424242, 15)]
        finally:
            dashboard.close()

    def test_stopping_an_idle_batch_is_refused(self, runs):
        dashboard = serve(runs, port=0, interval=1.0, allow_stop=True)
        try:
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

    def test_the_health_endpoint_reports_the_stop_capability(self, runs):
        dashboard = serve(runs, port=0, interval=1.0, allow_stop=False)
        try:
            status, body, _headers = self.get(dashboard.url + "api/health")
            assert status == 200
            assert json.loads(body) == {"status": "ok", "schema": SCHEMA, "stop": False}
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
            {"allow_stop": "yes"},
        ],
    )
    def test_invalid_settings_are_refused(self, tmp_path, kwargs):
        with pytest.raises(ValueError):
            serve(tmp_path, **kwargs)


# --------------------------------------------------------------------------
# The command line.
# --------------------------------------------------------------------------


def test_the_web_command_replaces_the_in_process_flag():
    parser = _parser()

    args = parser.parse_args(["web", "--port", "0"])
    assert args.port == 0
    assert args.host == "127.0.0.1"
    assert args.read_only is False

    with pytest.raises(SystemExit):
        parser.parse_args(["run", "pkg.mod:train", "cfg.yaml", "--web"])


def test_a_sleeping_dashboard_can_be_closed_while_it_waits(tmp_path):
    dashboard = serve(tmp_path, port=0, interval=1.0)
    try:
        dashboard.close()
    finally:
        dashboard.close()
