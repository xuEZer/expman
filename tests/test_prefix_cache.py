import random
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from expman import Batch, Pipeline, RunContext, Stage, Status
from expman.dependencies import ConfigurationReads, matches
from expman.frozen import FrozenDict
from expman.storage import PickleSerializer, RecoveryWarning, read_record


class PrefixCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def batch(self, stages, yaml):
        path = self.root / "config.yaml"
        path.write_text(yaml)
        return Batch(Pipeline(stages), path, output_dir=self.root / "batch")

    def test_prefix_restores_state_rng_metrics_and_stops_at_first_miss(self):
        calls = [0, 0, 0]

        class Prepare(Stage):
            def process(self, data, ctx):
                calls[0] += 1
                ctx.state["draw"] = random.random()
                ctx.log_metrics({"train": {"loss": 0.5}}, step=0)
                return [ctx.cfg["data"]]

        class Work(Stage):
            def process(self, data, ctx):
                calls[1] += 1
                return (data, ctx.state["draw"], random.random(), ctx.cfg["model"])

        class Finish(Stage):
            def process(self, data, ctx):
                calls[2] += 1
                return data

        batch = self.batch([Prepare, Work, Finish], "data: 1\nmodel: !choice [A, B]")
        results = batch.run(progress=False)
        self.assertTrue(all(r.status is Status.SUCCEEDED for r in results))
        self.assertEqual(calls, [1, 2, 2])
        self.assertEqual(results[0].output[:3], results[1].output[:3])
        refs = [
            read_record(e._store.stage_dir((0,)) / "completed.pkl", PickleSerializer())
            for e in batch.experiments
        ]
        self.assertEqual(refs[0]["cache_ref"], refs[1]["cache_ref"])
        self.assertNotIn("state", refs[1])
        results[1].output[0].append(99)
        self.assertEqual(results[0].output[0], [1])
        with sqlite3.connect(batch.output_dir / "metrics.sqlite3") as db:
            self.assertEqual(
                db.execute("SELECT count(DISTINCT run_id) FROM metrics").fetchone()[0],
                2,
            )
        resumed = Batch.resume(Pipeline([Prepare, Work, Finish]), batch.output_dir)
        self.assertEqual(resumed.run(progress=False)[1].output[0], [1])
        self.assertEqual(calls, [1, 2, 2])

    def test_complete_prefix_skips_all_constructors(self):
        calls = []

        class Work(Stage):
            def __init__(self):
                super().__init__()
                calls.append(1)

            def process(self, data, ctx):
                ctx.state["count"] = ctx.state.get("count", 0) + 1
                return ctx.state["count"]

        batch = self.batch([Work, Work], "unused: !choice [1, 2]")
        results = batch.run(progress=False)
        self.assertEqual([result.output for result in results], [2, 2])
        self.assertEqual(len(calls), 2)
        for index in (0, 1):
            self.assertEqual(
                batch.experiments[0]._store.completed_reference((index,)),
                batch.experiments[1]._store.completed_reference((index,)),
            )

    def test_stage_zero_mismatch_and_seed_partition(self):
        for field in ("value", "seed"):
            with self.subTest(field=field):
                calls = []

                class Work(Stage):
                    def process(self, data, ctx, calls=calls):
                        calls.append(ctx.stage_id)
                        return ctx.cfg["value"]

                path = self.root / f"{field}.yaml"
                path.write_text(
                    f"{field}: !choice [1, 2]\n"
                    + ("value: 1" if field == "seed" else "")
                )
                batch = Batch(
                    Pipeline([Work, Work]), path, output_dir=self.root / field
                )
                batch.run(progress=False)
                self.assertEqual(calls, [0, 1, 0, 1])

    def test_nested_pipeline_reads_and_container_access_are_dependencies(self):
        calls = []

        class Inner(Stage):
            def process(self, data, ctx):
                ctx.cfg["data"]
                return 1

        class Outer(Stage):
            def process(self, data, ctx):
                calls.append(1)
                return Pipeline([Inner]).run(ctx=ctx)

        batch = self.batch([Outer], "data:\n  unused: !choice [1, 2]")
        self.assertTrue(
            all(r.status is Status.SUCCEEDED for r in batch.run(progress=False))
        )
        self.assertEqual(len(calls), 2)

    def test_returned_config_is_tracked_during_serialization(self):
        class ReturnConfig(Stage):
            def process(self, data, ctx):
                return ctx.cfg

        batch = self.batch([ReturnConfig], "value: !choice [1, 2]")
        self.assertEqual([r.output["value"] for r in batch.run(progress=False)], [1, 2])

    def test_checkpoint_resume_keeps_conservative_dependencies(self):
        calls = []

        class Work(Stage):
            def process(self, data, ctx):
                calls.append(ctx.cfg["value"])
                if ctx.checkpoint.step is None:
                    ctx.state["value"] = ctx.cfg["value"]
                    ctx.checkpoint.save(step=1)
                    raise KeyboardInterrupt
                return ctx.state["value"]

        batch = self.batch([Work], "value: !choice [1, 2]")
        with self.assertRaises(KeyboardInterrupt):
            batch.run(progress=False)
        resumed = Batch.resume(Pipeline([Work]), batch.output_dir)
        with self.assertRaises(KeyboardInterrupt):
            resumed.run(progress=False)
        results = Batch.resume(Pipeline([Work]), batch.output_dir).run(progress=False)
        self.assertEqual([r.output for r in results], [1, 2])
        self.assertEqual(calls.count(1), 2)
        self.assertEqual(calls.count(2), 2)

    def test_corrupt_optional_cache_recomputes_and_publication_failure_fails_stage(
        self,
    ):
        calls = []

        class Work(Stage):
            def process(self, data, ctx):
                calls.append(1)
                return 5

        batch = self.batch([Work], "unused: !choice [1, 2]")
        self.assertEqual(batch.experiments[0].run().status, Status.SUCCEEDED)
        snapshot = next((batch.output_dir / "cache").rglob("snapshot.pkl"))
        snapshot.write_bytes(b"broken")
        with self.assertWarns(RecoveryWarning):
            result = batch.experiments[1].run()
        self.assertEqual(result.status, Status.SUCCEEDED)
        self.assertEqual(len(calls), 2)
        with patch.object(
            batch.experiments[0]._store.shared, "publish", side_effect=OSError("disk")
        ):
            (batch.experiments[0]._store.stage_dir((0,)) / "completed.pkl").unlink()
            result = batch.experiments[0].run()
        self.assertEqual(result.status, Status.FAILED)


class ConfigurationReadTests(unittest.TestCase):
    def test_missing_checks_are_rejected_at_every_depth(self):
        ctx = RunContext(cfg={"nested": {"x": 1}})
        for cfg in (ctx.cfg, ctx.cfg["nested"]):
            with self.assertRaises(TypeError):
                cfg.get("x")
            with self.assertRaises(TypeError):
                _ = "x" in cfg
            with self.assertRaises(KeyError):
                cfg["missing"]

    def test_missing_list_and_container_reads_are_conservative(self):
        original = {"data": {"values": [1, 2]}}
        tracker = ConfigurationReads(original)
        cfg = FrozenDict(original, tracker=tracker)
        tracker.begin()
        self.assertEqual(cfg["data"]["values"][0], 1)
        with self.assertRaises(KeyError):
            cfg["missing"]
        dependencies = tracker.export()
        self.assertTrue(matches(dependencies, original))
        self.assertFalse(matches(dependencies, {"data": {"values": [1, 3]}}))
        self.assertFalse(matches(dependencies, {**original, "missing": None}))
