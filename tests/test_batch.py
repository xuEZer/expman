import tempfile
import unittest
import weakref
from pathlib import Path

from expman import (
    Batch,
    ExecutionEvent,
    Experiment,
    InMemoryRecorder,
    MetricEvent,
    Pipeline,
    ProgressEvent,
    RunContext,
    Stage,
    Status,
)


class Echo(Stage):
    def process(self, data, ctx):
        return ctx.cfg


class Payload:
    """Weakref-able stage output used to check what the Batch keeps alive."""

    def __init__(self, item):
        self.item = item
        self.data = bytearray(1024)


class BatchTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "experiment.yaml"

    def make_batch(self, pipeline, cfg, **kwargs):
        directory = tempfile.mkdtemp(dir=self.path.parent)
        return Batch(pipeline, cfg, output_dir=Path(directory) / "batch", **kwargs)

    def config(self, text="item: !choice [A, B, C]"):
        self.path.write_text(text, encoding="utf-8")
        return self.path

    def test_batch_expands_yaml_and_returns_config_order(self):
        batch = self.make_batch(Pipeline([Echo]), self.config())
        self.assertEqual(len(batch.experiments), 3)
        self.assertTrue(
            all(result.status is Status.PENDING for result in batch.results)
        )
        results = batch.run()
        self.assertEqual([result.output["item"] for result in results], ["A", "B", "C"])
        self.assertEqual(len({result.run_id for result in results}), 3)
        self.assertTrue(all(result.status is Status.SUCCEEDED for result in results))
        self.assertTrue(all(len(result.attempts) == 1 for result in results))

    def test_all_ordinary_errors_rejoin_queue_tail_once(self):
        order = []

        class Work(Stage):
            def process(self, data, ctx):
                item = ctx.cfg["item"]
                order.append((item, ctx.attempt))
                if item == "A" and ctx.attempt == 1:
                    raise MemoryError("temporary failure")
                if item == "B":
                    raise ValueError("persistent failure")
                return item

        results = self.make_batch(Pipeline([Work]), self.config()).run()
        self.assertEqual(order, [("A", 1), ("B", 1), ("C", 1), ("A", 2), ("B", 2)])
        self.assertEqual(
            [result.status for result in results],
            [Status.SUCCEEDED, Status.FAILED, Status.SUCCEEDED],
        )
        self.assertEqual(results[0].output, "A")
        self.assertIsNone(results[1].output)
        self.assertEqual(results[0].attempts[0].error_type, "MemoryError")
        self.assertEqual(results[1].attempts[1].error_message, "persistent failure")
        self.assertEqual(
            results[0].duration_seconds,
            sum(item.duration_seconds for item in results[0].attempts),
        )

    def test_retry_reuses_completed_stage_state_and_output(self):
        starts = []
        instances = []

        class Prepare(Stage):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def process(self, data, ctx):
                instances.append(self)
                self.calls += 1
                starts.append(
                    (
                        ctx.cfg["item"],
                        ctx.attempt,
                        data,
                        self.calls,
                        list(ctx.cfg["values"]),
                    )
                )
                ctx.state["values"] = [*ctx.cfg["values"], 99]
                return "prepared"

        class Work(Stage):
            def process(self, data, ctx):
                if ctx.cfg["item"] == "A" and ctx.attempt == 1:
                    raise RuntimeError("retry")
                return (data, ctx.state["values"])

        batch = self.make_batch(
            Pipeline([Prepare, Work]), self.config("item: !choice [A, B]\nvalues: [1]")
        )
        results = batch.run()
        self.assertEqual(
            starts,
            [("A", 1, None, 1, [1]), ("B", 1, None, 1, [1])],
        )
        self.assertEqual(len({id(stage) for stage in instances}), 2)
        self.assertTrue(
            all(experiment.cfg["values"] == [1] for experiment in batch.experiments)
        )
        self.assertEqual(results[0].output, ("prepared", [1, 99]))

    def test_experiment_snapshots_config_and_attempt_results(self):
        cfg = {"items": [1]}
        experiment = Experiment(
            Pipeline([Echo]),
            cfg,
            run_id="stable",
            output_dir=self.path.parent / "stable",
        )
        cfg["items"].append(2)
        experiment.cfg["items"].append(3)
        first = experiment.run()
        snapshot = experiment.result
        with self.assertRaises(AttributeError):
            first.output["items"].append(4)
        second = experiment.run()
        self.assertEqual(second.output, {"items": [1]})
        self.assertEqual((first.run_id, second.run_id), ("stable", "stable"))
        self.assertEqual((first.attempt, second.attempt), (1, 2))
        self.assertEqual(len(snapshot.attempts), 1)
        self.assertEqual(len(experiment.result.attempts), 2)

    def test_construction_errors_receive_same_retry_budget(self):
        calls = []

        class Broken(Stage):
            def __init__(self):
                calls.append("init")
                raise RuntimeError("cannot initialize")

            def process(self, data, ctx):
                raise AssertionError("unreachable")

        batch = self.make_batch(Pipeline([Broken]), {})
        result = batch.run()[0]
        self.assertEqual(calls, ["init", "init"])
        self.assertEqual(result.status, Status.FAILED)
        finished = [
            event for event in batch.recorder.events if event.status is Status.FAILED
        ]
        self.assertEqual(
            [event.kind for event in finished], ["pipeline", "experiment"] * 2
        )

    def test_retry_budget_can_be_disabled_or_increased(self):
        class Fail(Stage):
            def process(self, data, ctx):
                raise RuntimeError("failure")

        for retries in (0, 1, 2):
            with self.subTest(retries=retries):
                result = self.make_batch(
                    Pipeline([Fail]), {}, max_retries=retries
                ).run()[0]
                self.assertEqual(len(result.attempts), retries + 1)

    def test_cancellation_stops_batch_and_retains_partial_results(self):
        for error_type in (KeyboardInterrupt, SystemExit):
            calls = []

            class Work(Stage):
                def process(self, data, ctx, *, _calls=calls, _error_type=error_type):
                    item = ctx.cfg["item"]
                    _calls.append(item)
                    if item == "B":
                        raise _error_type()
                    return item

            batch = self.make_batch(Pipeline([Work]), self.config())
            with self.subTest(error=error_type), self.assertRaises(error_type):
                batch.run()
            self.assertEqual(calls, ["A", "B"])
            self.assertEqual(
                [result.status for result in batch.results],
                [Status.SUCCEEDED, Status.CANCELLED, Status.PENDING],
            )
            self.assertEqual(len(batch.results[1].attempts), 1)
            self.assertEqual(batch.recorder.events[-1].status, Status.CANCELLED)

    def test_events_have_stable_run_id_and_attempt_number(self):
        contexts = []

        class Report(Stage):
            def process(self, data, ctx):
                contexts.append(ctx)
                ctx.report_metric("loss", 0.5)
                ctx.report_progress(1, total=2)
                if ctx.attempt == 1:
                    raise RuntimeError("retry")
                return "ok"

        recorder = InMemoryRecorder()
        batch = self.make_batch(
            Pipeline([Report]), {"full": {"config": 1}}, recorder=recorder
        )
        result = batch.run()[0]
        self.assertTrue(all(event.run_id == result.run_id for event in recorder.events))
        self.assertEqual({event.attempt for event in recorder.events}, {1, 2})
        for attempt in (1, 2):
            events = [event for event in recorder.events if event.attempt == attempt]
            experiment, pipeline, stage = events[:3]
            self.assertEqual(
                (experiment.kind, pipeline.kind, stage.kind),
                ("experiment", "pipeline", "stage"),
            )
            self.assertIsNone(experiment.parent_id)
            self.assertEqual(pipeline.parent_id, experiment.execution_id)
            self.assertEqual(stage.parent_id, pipeline.execution_id)
            self.assertIsInstance(events[3], MetricEvent)
            self.assertIsInstance(events[4], ProgressEvent)
            self.assertEqual(events[3].execution_id, stage.execution_id)
        self.assertIsNot(contexts[0], contexts[1])
        self.assertIsNot(contexts[0].cfg, contexts[1].cfg)
        self.assertEqual(contexts[0].cfg, {"full": {"config": 1}})
        starts = [
            event
            for event in recorder.events
            if isinstance(event, ExecutionEvent) and event.status is Status.RUNNING
        ]
        self.assertEqual(len({event.execution_id for event in starts}), 6)

    def test_failed_attempt_does_not_retain_stage_via_traceback(self):
        refs = []

        class Fail(Stage):
            def process(self, data, ctx):
                refs.append(weakref.ref(self))
                self.cycle = self
                raise RuntimeError("failure")

        result = self.make_batch(Pipeline([Fail]), {}).run()[0]
        self.assertTrue(all(reference() is None for reference in refs))
        self.assertEqual(result.attempts[0].error_message, "failure")

    def test_completed_attempt_does_not_retain_its_output(self):
        references = []

        class Produce(Stage):
            def process(self, data, ctx):
                output = Payload(ctx.cfg["item"])
                references.append(weakref.ref(output))
                return output

        batch = self.make_batch(
            Pipeline([Produce]), self.config("item: !choice [A, B]")
        )
        results = batch.run()
        self.assertEqual([result.output.item for result in results], ["A", "B"])
        self.assertTrue(all(reference() is None for reference in references))
        self.assertEqual(
            [attempt._output for result in results for attempt in result.attempts],
            [None, None],
        )
        self.assertEqual(
            [attempt.output.item for result in results for attempt in result.attempts],
            ["A", "B"],
        )

    def test_resumed_batch_reads_outputs_instead_of_retaining_them(self):
        batch = self.make_batch(Pipeline([Echo]), self.config("item: !choice [A, B]"))
        batch.run()
        resumed = Batch.resume(Pipeline([Echo]), batch.output_dir)
        attempts = [result.attempts[-1] for result in resumed.results]
        self.assertEqual([attempt._output for attempt in attempts], [None, None])
        self.assertTrue(all(attempt.output_source is not None for attempt in attempts))
        self.assertEqual([attempt.output["item"] for attempt in attempts], ["A", "B"])

    def test_error_message_failure_does_not_break_queue(self):
        class UnprintableError(Exception):
            def __str__(self):
                raise ValueError("cannot render")

        class Fail(Stage):
            def process(self, data, ctx):
                raise UnprintableError()

        result = self.make_batch(Pipeline([Fail]), {}).run()[0]
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(
            result.attempts[-1].error_message, "<exception message unavailable>"
        )

    def test_recorder_failure_does_not_trigger_retry(self):
        class BrokenRecorder:
            def record(self, event):
                raise OSError("unavailable")

        with self.assertLogs("expman.context", level="WARNING"):
            result = self.make_batch(
                Pipeline([Echo]), {}, recorder=BrokenRecorder()
            ).run()[0]
        self.assertEqual(result.status, Status.SUCCEEDED)
        self.assertEqual(len(result.attempts), 1)

    def test_one_concrete_config_and_empty_pipeline(self):
        batch = self.make_batch(Pipeline([]), {"seed": 42})
        result = batch.run()[0]
        self.assertEqual(result.status, Status.SUCCEEDED)
        self.assertIsNone(result.output)
        with self.assertRaises(RuntimeError):
            batch.run()

    def test_validation(self):
        for value in (-1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.make_batch(Pipeline([]), {}, max_retries=value)
        with self.assertRaises(TypeError):
            self.make_batch([], {})
        with self.assertRaises(TypeError):
            self.make_batch(Pipeline([]), [])
        with self.assertRaises(TypeError):
            Experiment(Pipeline([]), [])
        with self.assertRaises(ValueError):
            Experiment(Pipeline([]), {}, run_id=" ")
        with self.assertRaises(ValueError):
            RunContext(attempt=0)


if __name__ == "__main__":
    unittest.main()
