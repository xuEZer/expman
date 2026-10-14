"""Stage explicit checkpoint API: resume_range, checkpoint and restore."""

import random
from pathlib import Path

import pytest

from expman import Pipeline, Stage
from expman.core.context import RunContext
from expman.core.randomness import RandomStateManager
from expman.storage.base import RunStore, read_record, write_record


def managed_context(tmp_path: Path, cfg=None, seed=42):
    """A RunContext whose store continues one seeded random stream."""
    store = RunStore(tmp_path)
    rng = RandomStateManager(seed)
    initial = tmp_path / "rng_initial.pkl"
    if initial.exists():
        rng.restore(read_record(initial, store.serializer))
    else:
        rng.seed()
        write_record(initial, rng.capture(), store.serializer)
    store.rng = rng
    return RunContext(cfg={} if cfg is None else cfg, _store=store), store


def test_explicit_checkpoint_saves_user_state(tmp_path):
    """process() Stages can checkpoint arbitrary state at chosen positions."""

    class Train(Stage):
        def process(self, data, ctx):
            state = ctx.checkpoint_state
            if state is None:
                state = {"epoch": 0, "weights": [0.0]}
            for epoch in ctx.resume_range(0, 4):
                state["epoch"] = epoch + 1
                state["weights"].append(epoch)
                ctx.checkpoint(epoch + 1, state)
            return state

    ctx, store = managed_context(tmp_path)
    assert Pipeline([Train]).run(ctx=ctx) == {"epoch": 4, "weights": [0.0, 0, 1, 2, 3]}
    completed = store.completed((0,))
    assert completed["state"] == {"epoch": 4, "weights": [0.0, 0, 1, 2, 3]}


def test_explicit_checkpoint_resumes_from_saved_step(tmp_path):
    """A process() Stage resumes from the latest explicit checkpoint."""

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
                if type(self).fail_at is not None and epoch >= type(self).fail_at:
                    raise RuntimeError("interrupt")
            return state

    Train.fail_at = 2
    Train.seen = []
    ctx, store = managed_context(tmp_path)
    with pytest.raises(RuntimeError):
        Pipeline([Train]).run(ctx=ctx)
    assert Train.seen == [0, 1, 2]
    record = store.latest((0,))
    assert record["step"] == 3
    assert record["state"] == {"epoch": 3, "seen": [0, 1, 2]}

    Train.fail_at = None
    Train.seen = []
    ctx, _ = managed_context(tmp_path)
    result = Pipeline([Train]).run(ctx=ctx)
    assert result == {"epoch": 4, "seen": [0, 1, 2, 3]}
    assert Train.seen == [3]


def test_explicit_checkpoint_wraps_non_dict_state(tmp_path):
    """Non-dictionary state is wrapped transparently and unwrapped on resume."""

    class Counter(Stage):
        fail_at = 1
        seen = []

        def process(self, data, ctx):
            state = ctx.checkpoint_state
            if state is None:
                state = [0]
            for epoch in ctx.resume_range(0, 3):
                type(self).seen.append(epoch)
                state[0] += 1
                ctx.checkpoint(epoch + 1, state)
                if type(self).fail_at is not None and epoch >= type(self).fail_at:
                    raise RuntimeError("interrupt")
            return state

    Counter.fail_at = 1
    Counter.seen = []
    ctx, store = managed_context(tmp_path)
    with pytest.raises(RuntimeError):
        Pipeline([Counter]).run(ctx=ctx)
    record = store.latest((0,))
    assert record["state"] == {"__payload__": [2]}

    Counter.fail_at = None
    Counter.seen = []
    ctx, _ = managed_context(tmp_path)
    result = Pipeline([Counter]).run(ctx=ctx)
    assert result == [3]
    assert Counter.seen == [2]


def test_explicit_checkpoint_restores_random_state(tmp_path):
    """Resuming a process() Stage rewinds the random stream."""

    class Sampler(Stage):
        fail = True

        def process(self, data, ctx):
            state = ctx.checkpoint_state
            if state is None:
                state = []
            for epoch in ctx.resume_range(0, 4):
                state.append(random.random())
                ctx.checkpoint(epoch + 1, state)
                if epoch == 3 and type(self).fail:
                    random.random()  # consumed after the last checkpoint
                    type(self).fail = False
                    raise RuntimeError("interrupt")
            return state

    ctx, _ = managed_context(tmp_path)
    with pytest.raises(RuntimeError):
        Pipeline([Sampler]).run(ctx=ctx)
    ctx, _ = managed_context(tmp_path)
    output = Pipeline([Sampler]).run(ctx=ctx)
    expected = random.Random(42)
    assert output == [expected.random() for _ in range(4)]


def test_resume_range_is_empty_when_checkpoint_is_beyond_stop():
    """resume_range returns nothing if the checkpoint already reached the end."""
    ctx = RunContext(_checkpoint_record={"step": 5})
    assert list(ctx.resume_range(0, 2)) == []


def test_resume_range_resumes_from_the_recorded_step():
    """resume_range starts from max(start, the recorded step)."""
    ctx = RunContext(_checkpoint_record={"step": 2})
    assert list(ctx.resume_range(0, 5)) == [2, 3, 4]
    assert list(ctx.resume_range(3, 5)) == [3, 4]


def test_checkpoint_requires_managed_context():
    """checkpoint() outside a managed run raises RuntimeError."""

    class Train(Stage):
        def process(self, data, ctx):
            ctx.checkpoint(1, {})
            return data

    with pytest.raises(RuntimeError):
        Train().run(None)


def test_stage_without_process_is_rejected():
    """A Stage must override process(); loop() is no longer accepted."""

    class Empty(Stage):
        pass

    with pytest.raises(TypeError):
        Empty()


def test_bare_run_works_for_process_stages():
    """A process Stage can be run without a managed context."""

    class Double(Stage):
        def process(self, data, ctx):
            return data * 2

    assert Double().run(3) == 6
