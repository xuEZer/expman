from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from batch_helpers import make_batch, restore_batch

from expman import Pipeline, Stage, Status
from expman.core.events import ExecutionEvent, ProgressEvent
from expman.estimation.duration_model import StageDuration
from expman.observation.snapshot import _remaining
from expman.resources.devices import MemoryObservationError
from expman.scheduling._process import Worker
from expman.scheduling.scheduler import GpuScheduler


class First(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {}

    def process(self, data, ctx) -> None:
        pass


class Second(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {"item": True}

    def process(self, data, ctx) -> None:
        pass


def test_eta_counts_only_unmaterialized_stage_parameter_groups(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("device: [0]\nseed: 0\nitem: 0\ngrid:\n  item: [0, 1, 2]\n")
    batch = make_batch(Pipeline([First, Second]), cfg, output_dir=tmp_path / "batch")
    first = batch.experiments[0]
    scheduler = GpuScheduler(batch)
    # Three runs can share the queued first Stage. No run has reached the second.
    assert scheduler.stages.completion() == {0: (0, 1, 0, 1), 1: (0, 3, 0, 0)}
    batch.state.queue.remove(first.run_id)
    batch.state.active_gpu[first.run_id] = 0
    assert scheduler.stages.completion()[0] == (0, 1, 1, 0)
    del batch.state.active_gpu[first.run_id]
    batch.state.stage_history.extend(
        [
            {
                "run_id": first.run_id,
                "stage": 0,
                "status": Status.SUCCEEDED.value,
                "reused": False,
                "duration": 10.0,
                "stage_duration_seconds": 2.0,
                "dependencies": [],
            },
            {
                "run_id": first.run_id,
                "stage": 1,
                "status": Status.SUCCEEDED.value,
                "reused": False,
                "duration": 20.0,
                "stage_duration_seconds": 3.0,
                "dependencies": [(("item",), "sample")],
            },
        ]
    )
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
    assert scheduler.stages.completion() == {0: (1, 1, 0, 0), 1: (1, 3, 0, 0)}


RUNNING = StageDuration(30.0, 20.0, 40.0)
QUEUED = StageDuration(45.0, 30.0, 60.0)


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
    }
    scheduler = GpuScheduler(batch)
    monkeypatch.setattr(
        scheduler.stages,
        "duration",
        lambda _run_id, stage_index, **kwargs: (RUNNING, QUEUED)[stage_index],
    )
    return batch, scheduler, run_id


def test_eta_waits_for_the_running_prefix_even_with_an_idle_gpu(tmp_path, monkeypatch):
    batch, _scheduler, _run_id = _running_batch(
        tmp_path, monkeypatch, elapsed=10.0, devices=(0, 1)
    )

    estimate = batch.estimate()

    assert estimate.remaining_experiments == 2
    low, high = RUNNING.remaining(10.0, 0.8)
    assert estimate.lower_seconds == pytest.approx(low + QUEUED.lower)
    assert estimate.upper_seconds == pytest.approx(high + QUEUED.upper)
    assert batch.live.schedule_view["stages"][1]["groups"] == 1
    assert batch.live.schedule_view["stages"][1]["running"] == 0


def test_running_stage_keeps_a_positive_eta_through_overruns(tmp_path, monkeypatch):
    elapsed = 1557.8
    batch, _scheduler, _run_id = _running_batch(tmp_path, monkeypatch, elapsed=elapsed)
    estimate = batch.estimate()
    low, high = RUNNING.remaining(elapsed, 0.8)
    assert 0 < low < high
    assert estimate.lower_seconds == pytest.approx(low + QUEUED.lower)
    assert estimate.upper_seconds == pytest.approx(high + QUEUED.upper)
    assert estimate.remaining_experiments == 2
    assert batch.live.schedule_view["degraded"] is None
    assert _remaining(batch)["upper_seconds"] == pytest.approx(estimate.upper_seconds)
    batch._save_timing()
    assert (
        restore_batch(Pipeline([First, Second]), batch.output_dir)._last_estimate
        == estimate
    )


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


def test_running_peer_can_price_an_attempt_started_without_history(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("device: [0]\nseed: 0\nitem: 1\ngrid:\n  item: [1, 2]\n")
    batch = make_batch(Pipeline([Second]), cfg, output_dir=tmp_path / "batch")
    scheduler = GpuScheduler(batch)
    peer, active = batch.by_id
    for run_id in (peer, active):
        batch.state.start(run_id, 0, 0)
    assert not batch.estimate().known
    batch.state.release(peer)
    batch.state.stage_history.append(
        {
            "run_id": peer,
            "stage": 0,
            "status": "succeeded",
            "duration": 50.0,
            "stage_duration_seconds": 3.0,
        }
    )
    scheduler.stages.invalidate(0)
    # The other slot has been claimed, even before its first published worker sample.
    assert batch.estimate().known
    batch.live.running_info[active] = {"stage_index": 0, "duration": 55.0}
    eta = batch.estimate()
    assert eta.completed_samples == 1
    assert (eta.lower_seconds, eta.upper_seconds) == pytest.approx(
        StageDuration(50, 50, 50).remaining(55, 0.8)
    )


def test_one_unestimated_stage_clears_the_whole_batch_eta(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("device: [0]\nseed: 0\nitem: 1\ngrid:\n  item: [1, 2]\n")
    pipeline = Pipeline([Second, First, Second])
    batch = make_batch(pipeline, cfg, output_dir=tmp_path / "batch")
    scheduler = GpuScheduler(batch)
    predictions = [StageDuration(10, 10, 10)] * 3
    monkeypatch.setattr(
        scheduler.stages, "duration", lambda _, stage, **kw: predictions[stage]
    )
    assert batch.estimate().known
    batch._save_timing()
    predictions[1] = None
    eta = batch.estimate()
    assert (eta.lower_seconds, eta.upper_seconds) == (None, None)
    assert batch.live.schedule_view["stages"][1]["spans"] == 0
    remaining = _remaining(batch)
    assert remaining["lower_seconds"] is remaining["upper_seconds"] is None
    batch._save_timing()
    assert restore_batch(pipeline, batch.output_dir)._last_estimate is None


def test_first_stage_estimates_from_its_own_progress_after_resume(
    tmp_path, monkeypatch
):
    batch = make_batch(
        Pipeline([First]), {"device": [0], "seed": 0}, output_dir=tmp_path / "batch"
    )
    scheduler = GpuScheduler(batch)
    experiment = batch.experiments[0]
    run_id = experiment.run_id
    batch.state.start(run_id, 0, 0)
    worker = Worker(Mock(), experiment, 0, None, 0.0, channel=Mock())
    scheduler.workers[run_id] = worker
    stamp = datetime(2026, 1, 1, tzinfo=UTC)
    started = ExecutionEvent(
        run_id, "train", "pipeline", "stage", "Train", Status.RUNNING, stamp, stage_id=0
    )
    baseline = ProgressEvent(run_id, "train", 40, 100, "step", stamp, stage_id=0)

    def report(*events):
        worker.channel.drain.return_value = [
            {"type": "event", "event": e} for e in events
        ]
        worker.read_messages(batch.recorder)
        scheduler._account()

    nested = replace(baseline, execution_id="nested", completed=5)
    report(
        started,
        baseline,
        replace(started, execution_id="nested", parent_id="nested-pipeline"),
        nested,
        replace(nested, completed=6, timestamp=stamp + timedelta(seconds=1)),
    )
    assert not batch.estimate().known
    update = replace(baseline, completed=50, timestamp=stamp + timedelta(seconds=120))
    report(update, replace(update, timestamp=update.timestamp + timedelta(seconds=2)))
    monkeypatch.setattr(
        "expman.scheduling.parallel_estimation.time", update.timestamp.timestamp
    )
    eta = batch.estimate()
    assert eta.completed_samples == 0
    assert (eta.lower_seconds, eta.upper_seconds) == pytest.approx((300, 1200))
    report(
        replace(
            update,
            unit="epoch",
            completed=1,
            timestamp=update.timestamp + timedelta(seconds=3),
        )
    )
    assert not batch.estimate().known


def test_shared_running_prefix_unblocks_each_later_group_once(tmp_path, monkeypatch):
    batch, _scheduler, _run_id = _running_batch(
        tmp_path, monkeypatch, elapsed=10.0, devices=(0, 1), items=(0, 1, 2)
    )

    estimate = batch.estimate()

    # Three distinct second-Stage groups share one running first Stage. They
    # need two waves on two GPUs, both after that shared prefix finishes.
    assert estimate.remaining_experiments == 4
    low, high = RUNNING.remaining(10.0, 0.8)
    assert estimate.lower_seconds == pytest.approx(low + 2 * QUEUED.lower)
    assert estimate.upper_seconds == pytest.approx(high + 2 * QUEUED.upper)
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


def test_missing_elapsed_time_leaves_a_running_stage_unknown(tmp_path, monkeypatch):
    batch, _scheduler, run_id = _running_batch(tmp_path, monkeypatch, elapsed=2)
    del batch.live.running_info[run_id]["duration"]

    assert not batch.estimate().known


@pytest.mark.parametrize(
    ("elapsed", "durations"),
    [
        (None, [10, 12, 11, 18, 15, 13]),
        (1, [10, 12, 11, 18, 15, 13]),
        (20, [10, 12]),
    ],
    ids=["queued", "running", "overdue_with_two_samples"],
)
def test_explicit_coverage_reprices_the_stage_without_replacing_the_dashboard(
    tmp_path, elapsed, durations
):
    cfg = tmp_path / "cfg.yaml"
    items = list(range(len(durations) + 1))
    cfg.write_text(f"device: [0]\nseed: 0\nitem: 0\ngrid:\n  item: {items}\n")
    batch = make_batch(Pipeline([Second]), cfg, output_dir=tmp_path / "batch")
    for seconds in durations:
        batch.state.stage_history.append(
            {
                "run_id": batch.state.queue.popleft(),
                "stage": 0,
                "status": "succeeded",
                "duration": seconds,
            }
        )
    scheduler = GpuScheduler(batch)
    run_id = batch.state.queue[0]
    if elapsed is not None:
        batch.state.start(run_id, 0, 0)
        batch.live.running_info[run_id] = {
            "stage_index": 0,
            "duration": elapsed,
        }
    default = batch.estimate()
    view = batch.live.schedule_view
    narrow = batch.estimate(coverage=0.5)
    wide = batch.estimate(coverage=0.95)
    assert narrow.known and wide.known
    assert (
        wide.lower_seconds
        <= narrow.lower_seconds
        <= narrow.upper_seconds
        <= wide.upper_seconds
    )
    assert (
        wide.upper_seconds - wide.lower_seconds
        > narrow.upper_seconds - narrow.lower_seconds
    )
    expected = scheduler.stages.duration(
        run_id, 0, coverage=0.5 if elapsed is None else 0.8
    )
    span = (
        (expected.lower, expected.upper)
        if elapsed is None
        else expected.remaining(elapsed, 0.5)
    )
    assert (narrow.lower_seconds, narrow.upper_seconds) == pytest.approx(span)
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
