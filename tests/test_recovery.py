import json
import pickle
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

from expman import (
    Batch,
    Experiment,
    PickleSerializer,
    Pipeline,
    RecoveryWarning,
    RunContext,
    Stage,
    Status,
    StorageError,
)
from expman.storage import RunLock, read_record


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.output = self.root / "batch"

    def test_configuration_is_deeply_read_only_and_pickle_compatible(self):
        original = {"nested": {"layers": [1, {"size": 2}]}}
        ctx = RunContext(cfg=original)
        original["nested"]["layers"].append(3)
        self.assertEqual(ctx.cfg["nested"]["layers"], [1, {"size": 2}])
        with self.assertRaises(TypeError):
            ctx.cfg["nested"]["new"] = 1
        with self.assertRaises(TypeError):
            ctx.cfg["nested"]["layers"][0] = 4
        with self.assertRaises(TypeError):
            ctx.cfg["nested"]["layers"][1]["size"] = 3
        with self.assertRaises(AttributeError):
            ctx.cfg["nested"]["layers"].append(5)
        self.assertEqual(pickle.loads(pickle.dumps(ctx.cfg)), ctx.cfg)
        ctx.state["mutable"] = [1]
        ctx.state["mutable"].append(2)
        self.assertEqual(ctx.state, {"mutable": [1, 2]})

    def test_completed_stages_skip_construction_and_restore_state(self):
        constructors = []
        observed = []

        class Prepare(Stage):
            def __init__(self):
                super().__init__(name="duplicate")
                constructors.append("prepare")

            def process(self, data, ctx):
                ctx.state["items"] = [1]
                return {"input": 7}

        class Work(Stage):
            def __init__(self):
                super().__init__(name="duplicate")
                constructors.append("work")

            def process(self, data, ctx):
                observed.append(
                    (ctx.stage_id, ctx.attempt, data, list(ctx.state["items"]))
                )
                if ctx.attempt == 1:
                    ctx.state["items"].append(99)
                    raise ValueError("retry")
                return ctx.state

        batch = Batch(Pipeline([Prepare, Work]), {}, output_dir=self.output)
        result = batch.run()[0]
        self.assertEqual(constructors, ["prepare", "work", "work"])
        self.assertEqual(
            observed, [(1, 1, {"input": 7}, [1]), (1, 2, {"input": 7}, [1])]
        )
        self.assertEqual(result.output, {"items": [1]})
        stage_root = batch.experiments[0].output_dir / "stages"
        self.assertTrue((stage_root / "0/completed.pkl").exists())
        self.assertTrue((stage_root / "1/completed.pkl").exists())
        self.assertTrue(
            any(getattr(event, "reused", False) for event in batch.recorder.events)
        )

    def test_latest_checkpoint_restores_before_process_and_keeps_two(self):
        observed = []

        class Work(Stage):
            def process(self, data, ctx):
                observed.append((ctx.attempt, ctx.checkpoint.step, dict(ctx.state)))
                if ctx.attempt == 1:
                    for step in (1, 2, 3):
                        ctx.state["next"] = step
                        ctx.checkpoint.save(step=step)
                    ctx.state["unsaved"] = True
                    raise RuntimeError("retry")
                return ctx.state

        batch = Batch(Pipeline([Work]), {}, output_dir=self.output)
        result = batch.run()[0]
        self.assertEqual(observed, [(1, None, {}), (2, 3, {"next": 3})])
        self.assertEqual(result.output, {"next": 3})
        files = batch.experiments[0]._store.checkpoints((0,))
        self.assertEqual(len(files), 2)
        self.assertEqual([int(path.stem) for path in files], [3, 2])

    def test_corrupt_checkpoint_falls_back_then_to_stage_entry(self):
        class Prepare(Stage):
            def process(self, data, ctx):
                ctx.state["entry"] = 7
                return "input"

        class Work(Stage):
            def process(self, data, ctx):
                if ctx.attempt == 1:
                    for step in (1, 2):
                        ctx.state["next"] = step
                        ctx.checkpoint.save(step=step)
                    raise KeyboardInterrupt()
                return (data, ctx.checkpoint.step, ctx.state)

        for corrupt_count in (1, 2):
            output = self.root / str(corrupt_count)
            pipeline = Pipeline([Prepare, Work])
            batch = Batch(pipeline, {}, output_dir=output)
            with self.assertRaises(KeyboardInterrupt):
                batch.run()
            for path in batch.experiments[0]._store.checkpoints((1,))[:corrupt_count]:
                path.write_bytes(b"broken")
            resumed = Batch.resume(pipeline, output)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = resumed.run()[0]
            self.assertEqual(len(caught), corrupt_count)
            self.assertTrue(all(item.category is RecoveryWarning for item in caught))
            expected = (
                ("input", 1, {"entry": 7, "next": 1})
                if corrupt_count == 1
                else ("input", None, {"entry": 7})
            )
            self.assertEqual(result.output, expected)

    def test_snapshot_serialization_failure_is_a_stage_failure(self):
        class Work(Stage):
            def process(self, data, ctx):
                return (lambda: None) if ctx.attempt == 1 else {"ok": True}

        experiment = Experiment(Pipeline([Work]), {}, output_dir=self.output)
        failed = experiment.run()
        self.assertEqual(failed.status, Status.FAILED)
        self.assertEqual(failed.error_type, "StorageError")
        self.assertFalse((self.output / "stages/0/completed.pkl").exists())
        self.assertEqual(list(self.output.rglob(".writing-*")), [])
        self.assertEqual(experiment.run().output, {"ok": True})

    def test_checkpoint_write_failure_preserves_previous_checkpoint(self):
        class Work(Stage):
            def process(self, data, ctx):
                if ctx.attempt == 1:
                    ctx.state["next"] = 1
                    ctx.checkpoint.save(step=1)
                    ctx.state["bad"] = lambda: None
                    ctx.checkpoint.save(step=2)
                return ctx.state

        batch = Batch(Pipeline([Work]), {}, output_dir=self.output)
        result = batch.run()[0]
        self.assertEqual(
            [item.status for item in result.attempts], [Status.FAILED, Status.SUCCEEDED]
        )
        self.assertEqual(result.output, {"next": 1})
        self.assertEqual(len(batch.experiments[0]._store.checkpoints((0,))), 1)

    def test_interruptions_do_not_consume_failure_retry_budget(self):
        class Work(Stage):
            def process(self, data, ctx):
                if ctx.attempt == 1:
                    raise KeyboardInterrupt()
                if ctx.attempt == 2:
                    raise ValueError("one real failure")
                return "ok"

        pipeline = Pipeline([Work])
        batch = Batch(pipeline, {}, output_dir=self.output)
        with self.assertRaises(KeyboardInterrupt):
            batch.run()
        resumed = Batch.resume(pipeline, self.output)
        result = resumed.run()[0]
        self.assertEqual(result.run_id, batch.results[0].run_id)
        self.assertEqual(
            [item.status for item in result.attempts],
            [Status.CANCELLED, Status.FAILED, Status.SUCCEEDED],
        )

    def test_persisted_queue_keeps_retry_order(self):
        order = []

        class Work(Stage):
            def process(self, data, ctx):
                item = ctx.cfg["item"]
                order.append((item, ctx.attempt))
                if ctx.attempt == 1 and item == "A":
                    raise ValueError("retry")
                if ctx.attempt == 1 and item == "B":
                    raise KeyboardInterrupt()
                return item

        config = self.root / "experiment.yaml"
        config.write_text("item: !choice [A, B, C]")
        pipeline = Pipeline([Work])
        batch = Batch(pipeline, config, output_dir=self.output)
        ids = [item.run_id for item in batch.experiments]
        with self.assertRaises(KeyboardInterrupt):
            batch.run()
        config.unlink()
        resumed = Batch.resume(pipeline, self.output)
        results = resumed.run()
        self.assertEqual(order, [("A", 1), ("B", 1), ("B", 2), ("C", 1), ("A", 2)])
        self.assertEqual([item.run_id for item in results], ids)
        self.assertEqual([item.output for item in results], ["A", "B", "C"])
        finished = Batch.resume(pipeline, self.output)
        self.assertEqual([item.output for item in finished.run()], ["A", "B", "C"])
        self.assertEqual(len(order), 5)

    def test_changed_pipeline_and_stale_resume_are_rejected(self):
        class A(Stage):
            def process(self, data, ctx):
                return 1

        class B(Stage):
            def process(self, data, ctx):
                return 2

        pipeline = Pipeline([A, B])
        batch = Batch(pipeline, {}, output_dir=self.output)
        with self.assertRaises(StorageError):
            Batch.resume(Pipeline([B, A]), self.output)
        stale = Batch.resume(pipeline, self.output)
        batch.run()
        with self.assertRaises(StorageError):
            stale.run()

    def test_run_directory_is_not_overwritten_and_concurrent_run_is_rejected(self):
        pipeline = Pipeline([])
        batch = Batch(pipeline, {}, output_dir=self.output)
        with self.assertRaises(FileExistsError):
            Batch(pipeline, {}, output_dir=self.output)
        with RunLock(self.output), self.assertRaises(StorageError):
            batch.run()
        self.assertEqual(batch.run()[0].status, Status.SUCCEEDED)

    def test_nested_pipeline_invocations_have_distinct_positions(self):
        class Count(Stage):
            def process(self, data, ctx):
                ctx.state["count"] = ctx.state.get("count", 0) + 1
                return ctx.state["count"]

        class Outer(Stage):
            def process(self, data, ctx):
                inner = Pipeline([Count])
                inner.run(ctx=ctx)
                return inner.run(ctx=ctx)

        batch = Batch(Pipeline([Outer]), {}, output_dir=self.output)
        self.assertEqual(batch.run()[0].output, 2)
        store = batch.experiments[0]._store
        self.assertEqual(store.completed((0, 0, 0))["output"], 1)
        self.assertEqual(store.completed((0, 1, 0))["output"], 2)

    def test_checkpoint_restores_nested_pipeline_invocation_position(self):
        class Count(Stage):
            def process(self, data, ctx):
                ctx.state["count"] = ctx.state.get("count", 0) + 1
                return ctx.state["count"]

        class Outer(Stage):
            def process(self, data, ctx):
                for index in range(ctx.state.get("next", 0), 2):
                    Pipeline([Count]).run(ctx=ctx)
                    ctx.state["next"] = index + 1
                    ctx.checkpoint.save(step=index + 1)
                    if ctx.attempt == 1:
                        raise RuntimeError("retry")
                return ctx.state["count"]

        batch = Batch(Pipeline([Outer]), {}, output_dir=self.output)
        self.assertEqual(batch.run()[0].output, 2)

    def test_cross_process_sigint_and_abrupt_exit_recover(self):
        script = self.root / "driver.py"
        script.write_text("""
import json
import os
import signal
import sys
from pathlib import Path
from expman import Batch, Pipeline, Stage

root = Path(sys.argv[2])
class Prepare(Stage):
    def process(self, data, ctx):
        with (root / "calls").open("a") as stream:
            stream.write("prepare\\n")
        ctx.state["base"] = 10
        return 5

class Work(Stage):
    def process(self, data, ctx):
        start = ctx.state.get("next", 0)
        for index in range(start, 3):
            ctx.state["next"] = index + 1
            ctx.checkpoint.save(step=index + 1)
            if not (root / "interrupted").exists():
                (root / "interrupted").touch()
                if sys.argv[1] == "hard":
                    os._exit(23)
                signal.raise_signal(signal.SIGINT)
        return data + ctx.state["base"] + ctx.state["next"]

pipeline = Pipeline([Prepare, Work])
batch = Batch.resume(pipeline, root / "batch") if sys.argv[1] == "resume" else Batch(pipeline, {}, output_dir=root / "batch")
(root / "id").write_text(batch.experiments[0].run_id)
try:
    result = batch.run()[0]
except KeyboardInterrupt:
    sys.exit(10)
print(json.dumps({"run_id": result.run_id, "output": result.output,
                  "statuses": [attempt.status.value for attempt in result.attempts]}))
""")
        for mode, returncode in (("soft", 10), ("hard", 23)):
            root = self.root / mode
            root.mkdir()
            first = subprocess.run(
                [sys.executable, str(script), mode, str(root)],
                text=True,
                capture_output=True,
                timeout=20,
            )
            self.assertEqual(first.returncode, returncode, first.stderr)
            identity = (root / "id").read_text()
            second = subprocess.run(
                [sys.executable, str(script), "resume", str(root)],
                text=True,
                capture_output=True,
                timeout=20,
            )
            self.assertEqual(second.returncode, 0, second.stderr)
            result = json.loads(second.stdout)
            self.assertEqual(
                result,
                {
                    "run_id": identity,
                    "output": 18,
                    "statuses": ["cancelled", "succeeded"],
                },
            )
            self.assertEqual((root / "calls").read_text(), "prepare\n")

    def test_config_file_is_saved_once_and_stage_status_is_automatic(self):
        class Work(Stage):
            def process(self, data, ctx):
                if ctx.attempt == 1:
                    raise ValueError("retry")
                return "ok"

        batch = Batch(Pipeline([Work]), {"seed": 7}, output_dir=self.output)
        root = batch.experiments[0].output_dir
        before = (root / "config.pkl").stat().st_mtime_ns
        batch.run()
        self.assertEqual((root / "config.pkl").stat().st_mtime_ns, before)
        status = read_record(root / "stages/0/status.pkl", PickleSerializer())
        self.assertEqual(
            (status["stage_id"], status["attempt"], status["status"]),
            (0, 2, "succeeded"),
        )

    def test_custom_serializer_is_used_for_writing_and_resume(self):
        class PrefixedSerializer(PickleSerializer):
            def dump(self, value, stream):
                stream.write(b"EXPM")
                super().dump(value, stream)

            def load(self, stream):
                if stream.read(4) != b"EXPM":
                    raise ValueError("wrong serializer")
                return super().load(stream)

        class Work(Stage):
            def process(self, data, ctx):
                ctx.state["value"] = 42
                ctx.checkpoint.save(step=1)
                return ctx.state

        pipeline = Pipeline([Work])
        batch = Batch(
            pipeline, {}, output_dir=self.output, serializer=PrefixedSerializer()
        )
        self.assertEqual(batch.run()[0].output, {"value": 42})
        restored = Batch.resume(pipeline, self.output, serializer=PrefixedSerializer())
        self.assertEqual(restored.run()[0].output, {"value": 42})
        self.assertTrue(
            all(
                path.read_bytes().startswith(b"EXPM")
                for path in self.output.rglob("*.pkl")
            )
        )


if __name__ == "__main__":
    unittest.main()
