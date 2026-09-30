import json
import os
import socket
from time import perf_counter, sleep, time
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
import yaml

from expman import Batch, Pipeline, Stage, Status, devices
from expman._duration_model import StageDuration
from expman.devices import DeviceMemory, HostMemory
from expman.parallel_estimation import estimate_parallel
from expman.scheduling import GpuScheduler, Worker
from expman.webui import SCHEMA, build_snapshot, serve

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


@pytest.fixture(autouse=True)
def fake_devices(monkeypatch):
    monkeypatch.setattr(devices, "NvidiaMemory", Memory)
    monkeypatch.setattr(devices, "MeminfoMonitor", Host)
    monkeypatch.setattr(devices, "POLL_INTERVAL", 0.01)


def make_batch(root, *, count=2, delay=0.0):
    cfg = {"device": [0], "seed": 0, "markers": str(root), "delay": delay}
    path = root / "experiments.yaml"
    path.write_text(yaml.safe_dump(cfg) + f"item: !choice {list(range(count))}\n")
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


def test_a_snapshot_degrades_instead_of_raising():
    snapshot = build_snapshot(object())

    assert snapshot["schema"] == SCHEMA
    assert snapshot["degraded"].startswith("AttributeError")


class TestDashboard:
    def get(self, request):
        """Fetch a URL or a prepared request, treating an error as a response."""
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, response.read(), response.headers
        except HTTPError as error:
            return error.code, error.read(), error.headers

    def test_the_page_and_its_assets_are_served(self):
        dashboard = serve(object(), port=0, interval=1.0)
        try:
            dashboard.refresh()
            status, body, headers = self.get(dashboard.url)
            assert status == 200
            assert b"expman" in body
            assert headers["Content-Type"].startswith("text/html")
            status, _body, headers = self.get(dashboard.url + "static/app.js")
            assert status == 200
            assert headers["Content-Type"].startswith("text/javascript")
            status, _body, _headers = self.get(dashboard.url + "static/nope.js")
            assert status == 404
            status, _body, _headers = self.get(dashboard.url + "api/nope")
            assert status == 404
        finally:
            dashboard.close()

    def test_the_snapshot_endpoint_returns_the_published_payload(self):
        dashboard = serve(object(), port=0, interval=1.0)
        try:
            dashboard.refresh()
            status, body, headers = self.get(dashboard.url + "api/snapshot")
            assert status == 200
            assert headers["Content-Type"].startswith("application/json")
            payload = json.loads(body)
            assert payload["schema"] == SCHEMA
            assert "degraded" in payload
            status, body, _headers = self.get(dashboard.url + "api/health")
            assert status == 200
            assert json.loads(body)["stop"] is False
        finally:
            dashboard.close()

    def test_the_stream_pushes_a_frame(self):
        dashboard = serve(object(), port=0, interval=1.0)
        try:
            dashboard.refresh()
            with urlopen(dashboard.url + "api/stream", timeout=5) as response:
                assert response.readline().startswith(b"retry:")
                while True:
                    line = response.readline()
                    if line.startswith(b"data:"):
                        payload = json.loads(line[len(b"data:") :])
                        break
            assert payload["schema"] == SCHEMA
        finally:
            dashboard.close()

    def test_stopping_is_refused_unless_the_caller_allows_it(self):
        dashboard = serve(object(), port=0, interval=1.0)
        try:
            status, _body, _headers = self.get(
                Request(dashboard.url + "api/stop", method="POST")
            )
            assert status == 403
        finally:
            dashboard.close()

    def test_stopping_calls_back_once_allowed(self):
        calls = []
        dashboard = serve(
            object(),
            port=0,
            interval=1.0,
            allow_stop=True,
            on_stop=lambda: calls.append(1),
        )
        try:
            status, body, _headers = self.get(
                Request(dashboard.url + "api/stop", method="POST")
            )
            assert status == 200
            assert json.loads(body)["status"] == "stopping"
            for _ in range(50):
                if calls:
                    break
                sleep(0.05)
            assert len(calls) == 1
        finally:
            dashboard.close()

    def test_a_taken_port_falls_back_to_the_next_one(self):
        with socket.socket() as blocker:
            blocker.bind(("127.0.0.1", 0))
            blocker.listen(1)
            taken = blocker.getsockname()[1]
            dashboard = serve(object(), port=taken, interval=1.0)
            try:
                assert taken < dashboard.port < taken + 10
                assert self.get(dashboard.url)[0] == 200
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
    def test_invalid_settings_are_refused(self, kwargs):
        with pytest.raises(ValueError):
            serve(object(), **kwargs)
