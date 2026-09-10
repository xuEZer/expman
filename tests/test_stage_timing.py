import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from expman import Batch, Pipeline, Stage
from expman.storage import (
    Checkpoint,
    RecoveryWarning,
    RunStore,
    read_record,
    write_record,
)


class StageTimingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.now = 0.0
        clock = patch("expman.storage.perf_counter", lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)

    def test_interrupt_restores_time_with_progress_and_shared_source(self):
        owner = self

        class Work(Stage):
            def process(self, data, ctx):
                if ctx.checkpoint.step is None:
                    owner.now += 100
                    ctx.state["value"] = 10
                    ctx.checkpoint.save(step=10)
                    owner.now += 20
                    raise KeyboardInterrupt
                owner.now += 30
                return ctx.state["value"]

        batch = Batch(Pipeline([Work]), {}, output_dir=self.root / "batch")
        with self.assertRaises(KeyboardInterrupt):
            batch.run(progress=False)
        store = batch.experiments[0]._store
        self.assertEqual(store.latest((0,))["elapsed_seconds"], 100)
        self.now += 10000
        resumed = Batch.resume(Pipeline([Work]), batch.output_dir)
        self.assertEqual(resumed.run(progress=False)[0].output, 10)
        store = resumed.experiments[0]._store
        self.assertEqual(store.completed((0,))["elapsed_seconds"], 130)
        status = read_record(store.stage_dir((0,)) / "status.pkl", store.serializer)
        self.assertEqual(status["elapsed_seconds"], 130)
        self.assertEqual(store._timers, {})

    def test_shared_reuse_keeps_compute_duration_and_records_restore(self):
        owner = self

        class Work(Stage):
            def process(self, data, ctx):
                owner.now += 12
                return 42

        cfg = self.root / "cfg.yaml"
        cfg.write_text("unused: !choice [1, 2]")
        batch = Batch(Pipeline([Work]), cfg, output_dir=self.root / "batch")
        batch.run(progress=False)
        for experiment in batch.experiments:
            self.assertEqual(experiment._store.completed((0,))["elapsed_seconds"], 12)
        store = batch.experiments[1]._store
        status = read_record(store.stage_dir((0,)) / "status.pkl", store.serializer)
        self.assertTrue(status["reused"])
        self.assertEqual(status["elapsed_seconds"], 12)
        self.assertGreaterEqual(status["restore_seconds"], 0)

    def test_checkpoint_cutoffs_failure_and_corrupt_fallback(self):
        store = RunStore(self.root)
        store.start_timing((0,))
        checkpoint = Checkpoint(store, (0,), {})
        self.now = 5
        first = checkpoint.save(step=1)
        self.now = 9
        second = checkpoint.save(step=2)
        self.now = 12
        with (
            patch("expman.storage.write_record", side_effect=OSError("disk")),
            self.assertRaises(OSError),
        ):
            checkpoint.save(step=3)
        self.assertEqual(store.latest((0,))["elapsed_seconds"], 9)
        record = read_record(second, store.serializer)
        record["elapsed_seconds"] = float("nan")
        write_record(second, record, store.serializer)
        with self.assertWarns(RecoveryWarning):
            restored = store.latest((0,))
        self.assertEqual(restored["elapsed_seconds"], 5)
        self.now = 1000
        store.start_timing((0,), restored)
        self.now += 2
        self.assertEqual(store.elapsed((0,)), 7)
        self.assertTrue(first.exists())

    def test_legacy_timing_stays_unknown(self):
        store = RunStore(self.root)
        store.start_timing((0,), {"state": {}})
        self.now = 500
        checkpoint = Checkpoint(store, (0,), {})
        checkpoint.save()
        self.assertIsNone(store.latest((0,))["elapsed_seconds"])

    def test_nested_stage_clocks_are_independent(self):
        owner = self

        class Inner(Stage):
            def process(self, data, ctx):
                owner.now += 3
                ctx.checkpoint.save()
                return 1

        class Outer(Stage):
            def process(self, data, ctx):
                owner.now += 2
                result = Pipeline([Inner]).run(ctx=ctx)
                owner.now += 4
                return result

        batch = Batch(Pipeline([Outer]), {}, output_dir=self.root / "batch")
        batch.run(progress=False)
        store = batch.experiments[0]._store
        self.assertEqual(store.completed((0,))["elapsed_seconds"], 9)
        self.assertEqual(store.completed((0, 0, 0))["elapsed_seconds"], 3)
