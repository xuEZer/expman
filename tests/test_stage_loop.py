"""Stage loop API: init state capture, framework-driven resumable loop, break_loop."""

import random
from pathlib import Path

import pytest

from expman import Pipeline, Stage
from expman.context import RunContext
from expman.randomness import RandomStateManager
from expman.storage import RunStore, read_record, write_record


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


class Counter(Stage):
    """Counted loop that optionally interrupts at a given index."""

    fail_at = None
    seen = None

    def init(self, ctx):
        self.count = 0

    def loop(self, data, ctx, index, max_iter=4):
        type(self).seen.append(index)
        self.count += 1
        if type(self).fail_at is not None and index >= type(self).fail_at:
            raise RuntimeError("interrupt")
        return self.count


@pytest.fixture(autouse=True)
def reset_counter():
    Counter.fail_at = None
    Counter.seen = []
    yield


def test_init_state_variables_are_captured(tmp_path):
    class Work(Stage):
        def init(self, ctx):
            self.kept = 1
            self._private = "no"

        def process(self, data, ctx):
            self.later = "also no"
            return self.kept

    ctx, store = managed_context(tmp_path)
    assert Pipeline([Work]).run(ctx=ctx) == 1
    assert store.completed((0,))["state"] == {"kept": 1}


def test_loop_completes_and_leaves_no_checkpoint(tmp_path):
    ctx, store = managed_context(tmp_path)
    assert Pipeline([Counter]).run(ctx=ctx) == 4
    assert Counter.seen == [0, 1, 2, 3]
    assert store.checkpoints((0,)) == []
    assert store.completed((0,))["state"] == {"count": 4}


def test_resume_skips_init_and_continues_from_step(tmp_path):
    Counter.fail_at = 2
    ctx, store = managed_context(tmp_path)
    with pytest.raises(RuntimeError):
        Pipeline([Counter]).run(ctx=ctx)
    record = store.latest((0,))
    assert record["step"] == 2
    assert record["state"] == {"count": 2}
    Counter.fail_at = None
    ctx, _ = managed_context(tmp_path)
    assert Pipeline([Counter]).run(ctx=ctx) == 4
    # Positions 0 and 1 are skipped entirely: init never re-runs on a resume.
    assert Counter.seen == [0, 1, 2, 2, 3]


def test_resume_skips_init_entirely(tmp_path):
    class Traced(Stage):
        fail_at = 1

        def init(self, ctx):
            self.trace = ["init"]

        def loop(self, data, ctx, index, max_iter=2):
            self.trace.append(index)
            if type(self).fail_at is not None and index == type(self).fail_at:
                raise RuntimeError("interrupt")
            return self.trace

    ctx, _ = managed_context(tmp_path)
    with pytest.raises(RuntimeError):
        Pipeline([Traced]).run(ctx=ctx)
    # Attempt 1 ran init and iteration 0; the checkpoint holds ["init", 0].
    Traced.fail_at = None
    ctx, _ = managed_context(tmp_path)
    # init() running again would reset the trace to ["init"] before the loop.
    assert Pipeline([Traced]).run(ctx=ctx) == ["init", 0, 1]


def test_resume_rewinds_random_state(tmp_path):
    class Sampler(Stage):
        fail = True

        def init(self, ctx):
            self.samples = []

        def loop(self, data, ctx, index, max_iter=4):
            self.samples.append(random.random())
            if index == 3 and type(self).fail:
                random.random()  # Consumed after the last checkpoint.
                type(self).fail = False
                raise RuntimeError("interrupt")
            return self.samples

    ctx, _ = managed_context(tmp_path)
    with pytest.raises(RuntimeError):
        Pipeline([Sampler]).run(ctx=ctx)
    ctx, _ = managed_context(tmp_path)
    output = Pipeline([Sampler]).run(ctx=ctx)
    expected = random.Random(42)
    assert output == [expected.random() for _ in range(4)]


def test_max_iter_is_total_across_resumes(tmp_path):
    class Counted(Stage):
        fail_at = 1
        seen = []

        def init(self, ctx):
            self.count = 0

        def loop(self, data, ctx, index, max_iter=Stage.cfg("total")):
            type(self).seen.append(index)
            self.count += 1
            if type(self).fail_at is not None and index >= type(self).fail_at:
                raise RuntimeError("interrupt")
            return self.count

    ctx, _ = managed_context(tmp_path, cfg={"total": 3})
    with pytest.raises(RuntimeError):
        Pipeline([Counted]).run(ctx=ctx)
    Counted.fail_at = None
    ctx, _ = managed_context(tmp_path, cfg={"total": 3})
    assert Pipeline([Counted]).run(ctx=ctx) == 3
    assert Counted.seen == [0, 1, 1, 2]


def test_cfg_marker_missing_key_is_rejected(tmp_path):
    class Counted(Stage):
        def init(self, ctx):
            self.count = 0

        def loop(self, data, ctx, index, max_iter=Stage.cfg("total")):
            self.count += 1
            return self.count

    ctx, _ = managed_context(tmp_path)
    with pytest.raises(KeyError):
        Pipeline([Counted]).run(ctx=ctx)


def test_break_loop_stops_before_next_iteration(tmp_path):
    class EarlyStop(Stage):
        def init(self, ctx):
            self.count = 0

        def loop(self, data, ctx, index, max_iter=10):
            self.count += 1
            if index == 2:
                self.break_loop()
            return self.count

    ctx, store = managed_context(tmp_path)
    assert Pipeline([EarlyStop]).run(ctx=ctx) == 3
    # A break is a normal stop: the Stage completes and no checkpoint remains.
    assert store.checkpoints((0,)) == []
    assert store.completed((0,))["state"] == {"count": 3}


def test_loop_receives_upstream_data_and_skips_restored_positions(tmp_path):
    class Accumulate(Stage):
        fail_at = 2
        seen = None

        def init(self, ctx):
            self.total = 0

        def loop(self, data, ctx, index, max_iter=4):
            type(self).seen.append((index, data[index]))
            self.total += data[index]
            if type(self).fail_at is not None and index == type(self).fail_at:
                raise RuntimeError("interrupt")
            return self.total

    Accumulate.fail_at = 2
    Accumulate.seen = []
    ctx, _ = managed_context(tmp_path)
    with pytest.raises(RuntimeError):
        Pipeline([Accumulate]).run([10, 20, 30, 40], ctx=ctx)
    Accumulate.fail_at = None
    ctx, _ = managed_context(tmp_path)
    assert Pipeline([Accumulate]).run([10, 20, 30, 40], ctx=ctx) == 100
    assert Accumulate.seen == [(0, 10), (1, 20), (2, 30), (2, 30), (3, 40)]


def test_bare_run_degrades_silently():
    assert Counter().run(None) == 4
    assert Counter.seen == [0, 1, 2, 3]


def test_zero_max_iter_runs_nothing():
    class Empty(Stage):
        seen = None

        def init(self, ctx):
            self.count = 0

        def loop(self, data, ctx, index, max_iter=0):
            type(self).seen.append(index)
            return self.count

    assert Empty().run(None) is None
    assert Empty.seen is None  # loop() was never called.


def test_resume_after_final_step_returns_archived_result(tmp_path):
    class Finished(Stage):
        def init(self, ctx):
            self.count = 0

        def loop(self, data, ctx, index, max_iter=2):
            self.count += 1
            return self.count

    ctx, store = managed_context(tmp_path)

    def explode(*args, **kwargs):
        raise RuntimeError("crash after the last checkpoint")

    store.complete = explode
    with pytest.raises(RuntimeError):
        Pipeline([Finished]).run(ctx=ctx)
    ctx, _ = managed_context(tmp_path)
    # The loop already finished: the archived return value is the output.
    assert Pipeline([Finished]).run(ctx=ctx) == 2


def test_loop_signature_is_validated():
    with pytest.raises(TypeError):  # max_iter missing from the signature

        class Missing(Stage):
            def loop(self, data, ctx, index): ...

    with pytest.raises(TypeError):  # declared without a default value

        class NoDefault(Stage):
            def loop(self, data, ctx, index, max_iter): ...

    with pytest.raises(ValueError):  # negative literal

        class Negative(Stage):
            def loop(self, data, ctx, index, max_iter=-1): ...

    with pytest.raises(ValueError):  # bool is not an iteration count

        class Boolean(Stage):
            def loop(self, data, ctx, index, max_iter=True): ...

    with pytest.raises(TypeError):  # process() and loop() are mutually exclusive

        class Both(Stage):
            def loop(self, data, ctx, index, max_iter=1): ...

            def process(self, data, ctx):
                return data

    with pytest.raises(TypeError):  # a Stage must define one of the two

        class Neither(Stage):
            pass

        Neither()

    with pytest.raises(TypeError):  # Stage itself is not executable
        Stage()
