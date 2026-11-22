"""Stage explicit checkpoint API: resume_range, checkpoint and restore."""

import random

import pytest
from batch_helpers import make_batch

from expman import Pipeline, Stage, Status
from expman.core.context import RunContext
from expman.core.experiment import Experiment
from expman.scheduling.scheduler import GpuScheduler
from expman.storage.base import RecoveryWarning, StorageError, write_record
from expman.storage.run import RunStore


class Finish(Stage):
    def process(self, data, ctx):
        pass


def test_failed_checkpoint_write_preserves_last_saved_step(tmp_path):

    class Train(Stage):
        fail_at = 2
        seen = []

        def process(self, data, ctx):
            state = ctx.checkpoint_state
            if state is None:
                state = {"epoch": 0, "seen": []}
            for epoch in ctx.resume_range(0, 4):
                type(self).seen.append(epoch)
                state["epoch"] = epoch + 1
                state["seen"].append(epoch)
                ctx.checkpoint(epoch + 1, state)
                if epoch == type(self).fail_at:
                    ctx.checkpoint(epoch + 1, lambda: None)
            return state

    store = RunStore(tmp_path)
    pipeline = Pipeline([Train, Finish])
    with pytest.raises(StorageError, match="cannot save"):
        pipeline.run(ctx=RunContext(_store=store), stop_after=0)
    assert Train.seen == [0, 1, 2]
    record = store.latest((0,))
    assert record["step"] == 3
    assert record["state"] == {"epoch": 3, "seen": [0, 1, 2]}

    Train.fail_at = None
    Train.seen = []
    result = pipeline.run(ctx=RunContext(_store=RunStore(tmp_path)), stop_after=0)
    assert result == {"epoch": 4, "seen": [0, 1, 2, 3]}
    assert Train.seen == [3]
    assert store.completed((0,))["state"] == result
    assert not store.checkpoint_path((0,)).exists()


def test_checkpoint_with_null_state_resumes_with_no_steps_left(tmp_path):
    class Train(Stage):
        def process(self, data, ctx):
            assert list(ctx.resume_range(3, 5)) == [3, 4]
            for _ in ctx.resume_range(0, 1):
                ctx.checkpoint(1, None)
                raise RuntimeError("interrupt")
            return ctx.checkpoint_state

    pipeline = Pipeline([Train, Finish])
    with pytest.raises(RuntimeError, match="interrupt"):
        pipeline.run(ctx=RunContext(_store=RunStore(tmp_path)), stop_after=0)
    store = RunStore(tmp_path)
    assert pipeline.run(ctx=RunContext(_store=store), stop_after=0) is None
    assert store.completed((0,))["state"] is None


def test_explicit_checkpoint_restores_random_state(tmp_path):
    draws = []

    class Sampler(Stage):
        def process(self, data, ctx):
            for epoch in ctx.resume_range(0, 4):
                draws.append((epoch, random.random()))
                ctx.checkpoint(epoch + 1, None)
                if epoch == 1 and ctx.attempt == 1:
                    random.random()
                    raise RuntimeError("interrupt")

    run = Experiment(Pipeline([Sampler]), {"seed": 42}, output_dir=tmp_path / "run")
    assert run.run_stage(0, attempt=1).status is Status.FAILED
    assert run.run_stage(0, attempt=2).status is Status.SUCCEEDED
    expected = random.Random(42)
    assert draws == [(epoch, expected.random()) for epoch in range(4)]


def test_scheduled_stage_needs_only_its_predecessors_random_state(tmp_path):
    draws = []

    class Draw(Stage):
        def process(self, data, ctx):
            return random.random()

    class Collect(Stage):
        def process(self, data, ctx):
            draws.append((data, random.random()))

    experiment = Experiment(
        Pipeline([Draw, Collect]),
        {"seed": 42},
        output_dir=tmp_path / "run",
        _cache_root=tmp_path / "cache",
    )
    assert experiment.run_stage(0, attempt=1).status is Status.SUCCEEDED
    (experiment.output_dir / "rng_initial.pkl").unlink()
    random.random()
    assert experiment.run_stage(1, attempt=1).status is Status.SUCCEEDED
    expected = random.Random(42)
    assert draws == [(expected.random(), expected.random())]
    assert not (experiment.output_dir / "rng_initial.pkl").exists()


def test_direct_nested_stage_leaves_outer_checkpoint_for_retry(tmp_path):
    class Inner(Stage):
        def process(self, data, ctx):
            assert ctx.checkpoint_state is None
            with pytest.raises(RuntimeError, match="managed Stage"):
                ctx.checkpoint(1, "inner")
            return data + 1

    class Outer(Stage):
        def process(self, data, ctx):
            if ctx.checkpoint_state is None:
                ctx.checkpoint(1, 5)
                Inner().run(5, ctx)
                raise RuntimeError("outer interrupted")
            return Inner().run(ctx.checkpoint_state, ctx) * 2

    pipeline = Pipeline([Outer, Finish])
    store = RunStore(tmp_path)
    with pytest.raises(RuntimeError, match="outer interrupted"):
        pipeline.run(ctx=RunContext(_store=store))
    assert not store.has_completed((0,))
    assert store.latest((0,))["state"] == 5
    assert pipeline.run(ctx=RunContext(_store=RunStore(tmp_path)), stop_after=0) == 12
    assert store.completed((0,))["name"] == "Outer"


def test_run_resumes_its_own_checkpoint_before_reusing_shared_work(tmp_path):
    calls = []

    class LogMetric(Stage):
        def process(self, data, ctx):
            ctx.log_metrics({"result": 5})

    class Train(Stage):
        def process(self, data, ctx):
            calls.append((ctx.run_id, ctx.checkpoint_state))
            if len(calls) == 1:
                ctx.checkpoint(1, 5)
                raise RuntimeError("interrupted")
            Pipeline([LogMetric]).run(ctx=ctx)

    pipeline = Pipeline([Train])
    first, second, third = (
        Experiment(
            pipeline,
            {"seed": 42},
            run_id=name,
            output_dir=tmp_path / name,
            _cache_root=tmp_path / "cache",
            _metrics_path=tmp_path / "metrics.sqlite3",
        )
        for name in ("first", "second", "third")
    )
    assert first.run_stage(0, attempt=1).status is Status.FAILED
    assert second.run_stage(0, attempt=1).status is Status.SUCCEEDED
    assert first.run_stage(0, attempt=2).status is Status.SUCCEEDED
    assert third.run_stage(0, attempt=1).status is Status.SUCCEEDED
    assert calls == [("first", None), ("second", None), ("first", 5)]
    rows = third._metrics.snapshot("third", (0,))
    assert [value for _, _, _, value in rows] == [5]
    assert rows == second._metrics.snapshot("second", (0,))
    assert not list((tmp_path / "cache").rglob("snapshot.pkl"))


def test_both_cache_readers_compare_the_target_stage_declaration(tmp_path):
    class SumInputs(Stage):
        @classmethod
        def config_dependencies(cls, cfg):
            return {key: True for key in cfg if key.startswith("input_")}

        def process(self, data, ctx):
            return sum(ctx.cfg[key] for key in self.config_dependencies(ctx.cfg))

    class WriteTotal(Stage):
        def process(self, data, ctx):
            ctx.log_metrics({"total": data})

    path = tmp_path / "experiments.yaml"
    path.write_text("seed: 42\ndevice: [0]\ninput_x: 2\ncombines: [{}, {input_y: 5}]")
    batch = make_batch(
        Pipeline([SumInputs, WriteTotal]), path, output_dir=tmp_path / "batch"
    )
    first, second = batch.experiments
    assert first.run_stage(0, attempt=1).status is Status.SUCCEEDED
    assert first._store.completed((0,))["output"] == 2
    assert not GpuScheduler(batch)._reuse_stage(second.run_id)
    assert second.run_stage(0, attempt=1).status is Status.SUCCEEDED
    assert second._store.completed((0,))["output"] == 7


def test_invalid_timing_restarts_checkpoint_but_rejects_completed_stage(tmp_path):
    starts = []

    class Train(Stage):
        def process(self, data, ctx):
            starts.append(ctx.checkpoint_state)
            if len(starts) == 1:
                ctx.checkpoint(1, "training")
                raise RuntimeError("interrupted")
            return "done"

    store = RunStore(tmp_path)
    pipeline = Pipeline([Train, Finish])
    with pytest.raises(RuntimeError, match="interrupted"):
        pipeline.run(ctx=RunContext(_store=store))
    checkpoint = store.latest((0,))
    del checkpoint["elapsed_seconds"]
    write_record(store.checkpoint_path((0,)), checkpoint, store.serializer)
    with pytest.warns(RecoveryWarning, match="invalid cumulative stage duration"):
        assert pipeline.run(ctx=RunContext(_store=store), stop_after=0) == "done"
    assert starts == [None, None]

    completed = store.completed((0,))
    assert completed["elapsed_seconds"] >= 0
    completed["elapsed_seconds"] = None
    write_record(store.stage_dir((0,)) / "completed.pkl", completed, store.serializer)
    with pytest.raises(StorageError, match="invalid cumulative stage duration"):
        pipeline.run(ctx=RunContext(_store=store))
    assert len(starts) == 2
