import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from expman import Batch, Experiment, Pipeline, RunContext, Stage, Status, StorageError
from expman.metrics import MetricStore


class MetricsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.db = self.root / "metrics.sqlite3"
        self.ctx = RunContext(_metrics=MetricStore(self.db), _stage_path=(0,))

    def rows(self, path=None):
        with sqlite3.connect(path or self.db) as connection:
            return connection.execute(
                "SELECT run_id, stage_path, step, metric_path, value, attempt FROM metrics"
            ).fetchall()

    def test_nested_paths_partial_overwrite_and_summary(self):
        self.ctx.log_metrics({"train": {"mse": 1, "mae": 2}, "train/mse": 3}, step=0)
        replace(self.ctx, attempt=2).log_metrics({"train": {"mse": 0.5}}, step=0)
        self.ctx.log_metrics({"score": 1})
        self.ctx.log_metrics({"score": 2})
        rows = {tuple(json.loads(row[3])): row for row in self.rows()}
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows["train", "mse"][4:], (0.5, 2))
        self.assertEqual(rows["train", "mae"][4], 2)
        self.assertEqual(rows["train/mse",][4], 3)
        self.assertEqual(rows["score",][2], -1)
        self.assertEqual(rows["score",][4], 2)

    def test_validation_precedes_writes(self):
        self.ctx.log_metrics({"existing": 1})
        cyclic = {}
        cyclic["cycle"] = cyclic
        for invalid in [True, None, "text", float("nan"), float("inf"), 10**1000]:
            with self.subTest(invalid=type(invalid)), self.assertRaises(ValueError):
                self.ctx.log_metrics({"existing": 2, "nested": {"bad": invalid}})
        with self.assertRaises(ValueError):
            self.ctx.log_metrics(cyclic)
        for step in [-1, True, 0.5, 2**63]:
            with self.assertRaises(ValueError):
                self.ctx.log_metrics({"a": 1}, step=step)
        self.assertEqual(self.rows()[0][4], 1)
        with self.assertRaises(RuntimeError):
            RunContext().log_metrics({"a": 1})

    def test_transaction_rolls_back_all_updates(self):
        self.ctx.log_metrics({"a": 1})
        with sqlite3.connect(self.db) as connection:
            connection.execute("""CREATE TRIGGER reject_metric BEFORE INSERT ON metrics
                WHEN NEW.value = 99 BEGIN SELECT RAISE(ABORT, 'rejected'); END""")
        with self.assertRaises(StorageError):
            self.ctx.log_metrics({"a": 2, "b": 99})
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(self.rows()[0][4], 1)

    def test_nested_stage_and_experiment_identity(self):
        self.ctx.log_metrics({"a": 1}, step=0)
        replace(self.ctx, _stage_path=(0, 0, 0)).log_metrics({"a": 2}, step=0)
        replace(self.ctx, run_id="other").log_metrics({"a": 3}, step=0)
        self.assertEqual(len(self.rows()), 3)

    def test_interrupt_resume_overwrites_and_preserves_later_steps(self):
        class Train(Stage):
            def process(self, data, ctx):
                ctx.log_metrics({"train": {"loss": ctx.attempt}}, step=8)
                if ctx.attempt == 1:
                    ctx.checkpoint.save(step=8)
                    ctx.log_metrics({"train": {"loss": 10}}, step=10)
                    raise KeyboardInterrupt
                return 42

        pipeline = Pipeline([Train])
        batch = Batch(pipeline, {}, output_dir=self.root / "batch")
        with self.assertRaises(KeyboardInterrupt):
            batch.run()
        resumed = Batch.resume(pipeline, batch.output_dir)
        self.assertEqual(resumed.run()[0].output, 42)
        rows = self.rows(batch.output_dir / "metrics.sqlite3")
        self.assertEqual({row[2]: row[4:] for row in rows}, {8: (2, 2), 10: (10, 1)})
        self.assertEqual({row[0] for row in rows}, {batch.experiments[0].run_id})

    def test_storage_failure_retries_stage(self):
        class Train(Stage):
            def process(self, data, ctx):
                ctx.log_metrics({"loss": 1})

        batch = Batch(Pipeline([Train]), {}, output_dir=self.root / "batch")
        (batch.output_dir / "metrics.sqlite3").mkdir()
        result = batch.run()[0]
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(result.status, Status.FAILED)
        self.assertTrue(all(a.error_type == "StorageError" for a in result.attempts))
        self.assertFalse(list(batch.output_dir.rglob("completed.pkl")))

    def test_standalone_experiment(self):
        class Train(Stage):
            def process(self, data, ctx):
                ctx.log_metrics({"score": 3})

        experiment = Experiment(
            Pipeline([Train]), {}, output_dir=self.root / "experiment"
        )
        self.assertEqual(experiment.run().status, Status.SUCCEEDED)
        self.assertEqual(self.rows(experiment.output_dir / "metrics.sqlite3")[0][4], 3)
