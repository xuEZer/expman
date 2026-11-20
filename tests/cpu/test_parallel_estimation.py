from types import SimpleNamespace

import pytest
from batch_helpers import make_batch, restore_batch

from expman import Pipeline, Stage, Status
from expman.observation.snapshot import _remaining
from expman.resources.devices import MemoryObservationError
from expman.scheduling.scheduler import GpuScheduler
from expman.storage.base import read_record


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
    batch = make_batch(Pipeline([First, Second]), cfg, output_dir=tmp_path / "batch")
    first = batch.experiments[0]
    batch.state.queue.remove(first.run_id)
    batch.state.stage_history.extend(
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
    assert scheduler.stages.completion() == ((0, 1, 1), (1, 1, 3))


def _running_batch(tmp_path, monkeypatch, *, elapsed, devices=(0,), items=(0,)):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"device: {list(devices)}\nseed: 0\nitem: 0\ngrid:\n  item: {list(items)}\n"
    )
    batch = make_batch(Pipeline([First, Second]), cfg, output_dir=tmp_path / "batch")
    run_id = batch.experiments[0].run_id
    batch.state.queue.remove(run_id)
    batch.state.active_gpu[run_id] = devices[0]
    batch.live.running_info[run_id] = {
        "stage_index": 0,
        "duration": elapsed,
        "expected_lower_seconds": 20.0,
        "expected_upper_seconds": 40.0,
    }
    scheduler = GpuScheduler(batch)
    monkeypatch.setattr(
        scheduler.stages,
        "duration",
        lambda _run_id, _stage_index, **kwargs: SimpleNamespace(lower=30.0, upper=60.0),
    )
    return batch, scheduler, run_id


def test_eta_includes_running_experiments_later_stages(tmp_path, monkeypatch):
    batch, scheduler, run_id = _running_batch(tmp_path, monkeypatch, elapsed=10.0)

    estimate = batch.estimate()
    groups, active = scheduler.stages.remaining_groups([], batch.live.running_info)

    assert set(groups.values()) == {(run_id, 0), (run_id, 1)}
    assert len(active) == 1
    assert estimate.remaining_experiments == 2
    assert estimate.lower_seconds == 40.0
    assert estimate.upper_seconds == 90.0
    assert batch.live.schedule_view["stages"][1]["groups"] == 1
    assert batch.live.schedule_view["stages"][1]["running"] == 0


@pytest.mark.parametrize("elapsed", [40.0, 1557.8])
def test_overdue_running_stage_invalidates_cached_eta(tmp_path, monkeypatch, elapsed):
    batch, _scheduler, run_id = _running_batch(tmp_path, monkeypatch, elapsed=10.0)
    initial = batch.estimate()
    assert initial.upper_seconds is not None

    batch.live.running_info[run_id]["duration"] = elapsed
    estimate = batch.estimate()

    assert estimate.lower_seconds is None
    assert estimate.upper_seconds is None
    assert estimate.remaining_experiments == 2
    assert str(estimate) == "??:??:??～??:??:??"
    assert batch._snapshot_estimate() == estimate
    remaining = _remaining(batch)
    assert remaining["lower_seconds"] is None
    assert remaining["upper_seconds"] is None
    assert remaining["remaining_groups"] == 2
    view = batch.live.schedule_view
    assert "预测上界" in view["degraded"]
    assert view["makespan"] == {"lower_seconds": None, "upper_seconds": None}
    assert view["stages"][0]["spans"] == 0
    assert view["stages"][0]["unit_upper_seconds"] is None
    assert view["stages"][1]["unit_upper_seconds"] == 60.0
    batch._save_timing()
    timing = read_record(batch.output_dir / "timing.pkl", batch.store.serializer)
    assert timing["estimate"] is None
    assert (
        restore_batch(Pipeline([First, Second]), batch.output_dir)._last_estimate
        is None
    )


def test_running_stage_below_upper_bound_keeps_valid_residual_eta(
    tmp_path, monkeypatch
):
    batch, _scheduler, _run_id = _running_batch(tmp_path, monkeypatch, elapsed=39.0)

    estimate = batch.estimate()

    assert estimate.lower_seconds == 30.0
    assert estimate.upper_seconds == 61.0
    assert batch.live.schedule_view["degraded"] is None


def test_completed_batch_keeps_zero_remaining_eta(tmp_path, monkeypatch):
    batch, _scheduler, run_id = _running_batch(tmp_path, monkeypatch, elapsed=10.0)
    batch.estimate()
    batch.state.active_gpu.clear()
    batch.live.running_info.clear()
    batch.state.stage_progress[run_id] = 2

    estimate = batch.estimate()

    assert estimate.lower_seconds == 0.0
    assert estimate.upper_seconds == 0.0
    assert estimate.remaining_experiments == 0
    assert all(
        stage["groups"] == stage["running"] == 0
        for stage in batch.live.schedule_view["stages"].values()
    )
    assert batch.live.schedule_view["stages"][1]["unit_lower_seconds"] == 30.0


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
    assert batch.live.schedule_view["stages"][0]["groups"] == 1
    assert batch.live.schedule_view["stages"][1]["groups"] == 3


def test_unreadable_memory_keeps_idle_gpu_slots_closed(tmp_path, monkeypatch):
    batch, scheduler, _run_id = _running_batch(
        tmp_path, monkeypatch, elapsed=10.0, devices=(0, 1), items=(0, 1, 2)
    )

    def fail():
        raise MemoryObservationError("device counters unavailable")

    monkeypatch.setattr(scheduler.monitor, "sample", fail)
    with pytest.warns(RuntimeWarning, match="memory query failed"):
        scheduler._observe()
    assert batch.estimate().known
    assert batch.live.schedule_view["device_slots"] == {0: 1}


@pytest.mark.parametrize("active", [False, True], ids=["queued", "running"])
def test_explicit_coverage_reprices_the_stage_without_replacing_the_dashboard(
    tmp_path, active
):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "device: [0]\nseed: 0\nitem: 0\ngrid:\n  item: [0, 1, 2, 3, 4, 5, 6]\n"
    )
    batch = make_batch(Pipeline([Second]), cfg, output_dir=tmp_path / "batch")
    for seconds in [10, 12, 11, 18, 15, 13]:
        batch.state.stage_history.append(
            {
                "run_id": batch.state.queue.popleft(),
                "stage": 0,
                "status": "succeeded",
                "stage_duration_seconds": seconds,
            }
        )
    scheduler = GpuScheduler(batch)
    run_id = batch.state.queue[0]
    elapsed = 1.0 if active else 0.0
    if active:
        predicted = scheduler.stages.duration(run_id, 0)
        batch.state.start(run_id, 0, 0)
        batch.live.running_info[run_id] = {
            "stage_index": 0,
            "duration": elapsed,
            "expected_lower_seconds": predicted.lower,
            "expected_upper_seconds": predicted.upper,
        }
    default = batch.estimate()
    view = batch.live.schedule_view
    narrow = batch.estimate(coverage=0.5)
    wide = batch.estimate(coverage=0.95)
    assert narrow.known and wide.known
    assert (
        wide.upper_seconds - wide.lower_seconds
        > narrow.upper_seconds - narrow.lower_seconds
    )
    expected = scheduler.stages.duration(run_id, 0, coverage=0.5)
    assert narrow.lower_seconds == pytest.approx(expected.lower - elapsed)
    assert narrow.upper_seconds == pytest.approx(expected.upper - elapsed)
    assert batch._snapshot_estimate() == default
    assert batch.live.schedule_view is view


def test_greedy_scenarios_can_finish_in_the_opposite_order(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("device: [0, 1]\nseed: 0\nitem: 0\ngrid:\n  item: [0, 1, 2]\n")
    batch = make_batch(Pipeline([Second, Second]), cfg, output_dir=tmp_path / "batch")
    first, second, third = batch.by_id
    batch.state.stage_progress.update({second: 1, third: 1})
    spans = {
        (first, 0): (3, 5),
        (first, 1): (10, 10),
        (second, 1): (4, 6),
        (third, 1): (4, 4),
    }

    def predict(run_id, stage_index, *, coverage):
        lower, upper = spans[run_id, stage_index]
        return SimpleNamespace(lower=lower, upper=upper)

    scheduler = GpuScheduler(batch)
    monkeypatch.setattr(scheduler.stages, "duration", predict)
    estimate = batch.estimate()
    # In the second scenario the longer first Stage is placed early and its
    # 10-second successor starts sooner. Both completion times belong in the range.
    assert (estimate.lower_seconds, estimate.upper_seconds) == (15, 17)
