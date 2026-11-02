import io
from types import SimpleNamespace

import pytest

from expman import Batch, Pipeline, Stage, Status
from expman.scheduling.progress import BatchProgress
from expman.scheduling.scheduler import GpuScheduler
from expman.storage.base import read_record
from expman.webui.snapshot import _remaining


class First(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {}

    def process(self, data, ctx):
        return data


class Second(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {"item": True}

    def process(self, data, ctx):
        return ctx.cfg["item"]


def test_eta_counts_only_unmaterialized_stage_parameter_groups(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("device: [0]\nseed: 0\nitem: 0\ngrid:\n  item: [0, 1, 2]\n")
    batch = Batch(Pipeline([First, Second]), cfg, output_dir=tmp_path / "batch")
    first = batch.experiments[0]
    batch._queue.remove(first.run_id)
    batch._stage_history.extend(
        [
            {
                "run_id": first.run_id,
                "stage": 0,
                "status": Status.SUCCEEDED.value,
                "reused": False,
                "stage_duration_seconds": 10.0,
                "dependencies": [],
            },
            {
                "run_id": first.run_id,
                "stage": 1,
                "status": Status.SUCCEEDED.value,
                "reused": False,
                "stage_duration_seconds": 20.0,
                "dependencies": [(("item",), "sample")],
            },
        ]
    )
    scheduler = GpuScheduler(batch)

    estimate = batch.estimate()

    # Stage 0 has one config-independent cache group, already completed.
    # Stage 1 still needs one representative for item=1 and item=2.
    assert estimate.lower_seconds == 40.0
    assert estimate.upper_seconds == 40.0
    # One measured sample per Stage leaves a degenerate interval; the two
    # representatives then share the single available slot.
    assert str(estimate) == "00:00:00～00:00:01"
    assert estimate.completed_samples == 2
    assert estimate.remaining_experiments == 2
    assert batch._scheduler is scheduler
    progress = BatchProgress(batch, enabled=True, interval=1)
    progress.stream = io.StringIO()
    progress._render()
    assert "Stage0:1/1 Stage1:1/3" in progress.stream.getvalue()


def _running_batch(tmp_path, monkeypatch, *, elapsed, devices=(0,), items=(0,)):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"device: {list(devices)}\nseed: 0\nitem: 0\ngrid:\n  item: {list(items)}\n"
    )
    batch = Batch(Pipeline([First, Second]), cfg, output_dir=tmp_path / "batch")
    run_id = batch.experiments[0].run_id
    batch._queue.remove(run_id)
    batch._active_gpu[run_id] = devices[0]
    batch._gpu_running_info[run_id] = {
        "stage_index": 0,
        "duration": elapsed,
        "expected_lower_seconds": 20.0,
        "expected_upper_seconds": 40.0,
    }
    scheduler = GpuScheduler(batch)
    monkeypatch.setattr(
        scheduler,
        "_stage_duration",
        lambda _run_id, _stage_index: SimpleNamespace(lower=30.0, upper=60.0),
    )
    return batch, scheduler, run_id


def test_eta_includes_running_experiments_later_stages(tmp_path, monkeypatch):
    batch, scheduler, run_id = _running_batch(tmp_path, monkeypatch, elapsed=10.0)

    estimate = batch.estimate()
    groups, active = scheduler._remaining_stage_groups([], batch._gpu_running_info)

    assert set(groups.values()) == {(run_id, 0), (run_id, 1)}
    assert len(active) == 1
    assert estimate.remaining_experiments == 2
    assert estimate.lower_seconds == 40.0
    assert estimate.upper_seconds == 90.0
    assert batch._schedule_view["stages"][1]["groups"] == 1
    assert batch._schedule_view["stages"][1]["running"] == 0


@pytest.mark.parametrize("elapsed", [40.0, 1557.8])
def test_overdue_running_stage_invalidates_cached_eta(tmp_path, monkeypatch, elapsed):
    batch, _scheduler, run_id = _running_batch(tmp_path, monkeypatch, elapsed=10.0)
    initial = batch.estimate()
    assert initial.upper_seconds is not None

    batch._gpu_running_info[run_id]["duration"] = elapsed
    estimate = batch.estimate()

    assert estimate.lower_seconds is None
    assert estimate.upper_seconds is None
    assert estimate.remaining_experiments == 2
    assert str(estimate) == "??:??:??～??:??:??"
    assert batch._display_estimate() == estimate
    remaining = _remaining(batch)
    assert remaining["lower_seconds"] is None
    assert remaining["upper_seconds"] is None
    assert remaining["remaining_groups"] == 2
    view = batch._schedule_view
    assert "预测上界" in view["degraded"]
    assert view["makespan"] == {"lower_seconds": None, "upper_seconds": None}
    assert view["stages"][0]["spans"] == 0
    assert view["stages"][0]["unit_upper_seconds"] is None
    assert view["stages"][1]["unit_upper_seconds"] == 60.0
    batch._save_timing()
    timing = read_record(batch.output_dir / "timing.pkl", batch._serializer)
    assert timing["estimate"] is None
    assert (
        Batch.resume(Pipeline([First, Second]), batch.output_dir)._last_estimate is None
    )


def test_running_stage_below_upper_bound_keeps_valid_residual_eta(
    tmp_path, monkeypatch
):
    batch, _scheduler, _run_id = _running_batch(tmp_path, monkeypatch, elapsed=39.0)

    estimate = batch.estimate()

    assert estimate.lower_seconds == 30.0
    assert estimate.upper_seconds == 61.0
    assert batch._schedule_view["degraded"] is None


def test_completed_batch_keeps_zero_remaining_eta(tmp_path, monkeypatch):
    batch, _scheduler, run_id = _running_batch(tmp_path, monkeypatch, elapsed=100.0)
    batch._active_gpu.clear()
    batch._gpu_running_info.clear()
    batch._stage_progress[run_id] = 2

    estimate = batch.estimate()

    assert estimate.lower_seconds == 0.0
    assert estimate.upper_seconds == 0.0
    assert estimate.remaining_experiments == 0


def test_later_stage_waits_for_running_prefix_on_another_device(tmp_path, monkeypatch):
    batch, _scheduler, _run_id = _running_batch(
        tmp_path, monkeypatch, elapsed=10.0, devices=(0, 1)
    )

    estimate = batch.estimate()

    # A free second GPU cannot evaluate this run before its training finishes.
    assert estimate.lower_seconds == 40.0
    assert estimate.upper_seconds == 90.0


def test_shared_running_prefix_unblocks_each_later_group_once(tmp_path, monkeypatch):
    batch, _scheduler, _run_id = _running_batch(
        tmp_path, monkeypatch, elapsed=10.0, devices=(0, 1), items=(0, 1, 2)
    )

    estimate = batch.estimate()

    # Three distinct second-Stage groups share one running first Stage. They
    # need two waves on two GPUs, both after that shared prefix finishes.
    assert estimate.remaining_experiments == 4
    assert estimate.lower_seconds == 70.0
    assert estimate.upper_seconds == 150.0
    assert batch._schedule_view["stages"][0]["groups"] == 1
    assert batch._schedule_view["stages"][1]["groups"] == 3
