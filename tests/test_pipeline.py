import unittest
from unittest.mock import patch

from expman import (
    InMemoryRecorder,
    MetricEvent,
    Pipeline,
    ProgressEvent,
    RunContext,
    Stage,
    Status,
)


class Add(Stage[int, int]):
    def __init__(self, amount=1, **kwargs):
        super().__init__(**kwargs)
        self.amount = amount
        self.calls = 0

    def process(self, data, ctx):
        self.calls += 1
        return data + self.amount


def raising(error):
    class Raise(Stage):
        def process(self, data, ctx):
            raise error

    return Raise


class Report(Stage):
    def process(self, data, ctx):
        ctx.report_metric("loss", 0.5, step=2)
        ctx.report_progress(2, total=4, unit="batch")
        return data


class BrokenRecorder:
    def record(self, event):
        raise OSError("recorder unavailable")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.recorder = InMemoryRecorder()
        self.ctx = RunContext(recorder=self.recorder)

    def test_order_and_execution_hierarchy(self):
        class AddThree(Add):
            def __init__(self):
                super().__init__(3)

        class Double(Stage):
            def process(self, data, ctx):
                return data * 2

        result = Pipeline([AddThree, Double]).run(2, self.ctx)
        self.assertEqual(result, 10)
        events = self.recorder.events
        self.assertEqual(
            [event.status for event in events],
            [
                Status.RUNNING,
                Status.RUNNING,
                Status.SUCCEEDED,
                Status.RUNNING,
                Status.SUCCEEDED,
                Status.SUCCEEDED,
            ],
        )
        self.assertTrue(all(event.run_id == self.ctx.run_id for event in events))
        self.assertIsNone(events[0].parent_id)
        for index in (1, 2, 3, 4):
            self.assertEqual(events[index].parent_id, events[0].execution_id)
        self.assertEqual(events[0].execution_id, events[-1].execution_id)
        self.assertEqual(events[1].execution_id, events[2].execution_id)
        self.assertIsNone(self.ctx.execution_id)

    def test_empty_pipeline_preserves_input_identity(self):
        data = object()
        self.assertIs(Pipeline([]).run(data, self.ctx), data)
        self.assertEqual(len(self.recorder.events), 2)
        self.assertIsNone(Pipeline([]).run())

    def test_arbitrary_outputs_including_none_are_forwarded(self):
        class Clear(Stage):
            def process(self, data, ctx):
                return None

        class Wrap(Stage):
            def process(self, data, ctx):
                return {"value": data}

        self.assertEqual(Pipeline([Clear, Wrap]).run(object()), {"value": None})

    def test_failure_stops_downstream_and_preserves_exception(self):
        error = ValueError("bad input")
        calls = []

        class Downstream(Stage):
            def process(self, data, ctx):
                calls.append(data)

        with self.assertRaises(ValueError) as raised:
            Pipeline([raising(error), Downstream]).run(1, self.ctx)
        self.assertIs(raised.exception, error)
        self.assertEqual(calls, [])
        finished = self.recorder.events[-2:]
        self.assertEqual([event.status for event in finished], [Status.FAILED] * 2)
        for event in finished:
            self.assertEqual(event.error_type, "ValueError")
            self.assertEqual(event.error_message, "bad input")
            self.assertGreaterEqual(event.duration_seconds, 0)

    def test_cancellation_is_recorded_and_propagated(self):
        for error in (KeyboardInterrupt(), SystemExit(2)):
            with self.subTest(error=type(error).__name__):
                recorder = InMemoryRecorder()
                with self.assertRaises(type(error)) as raised:
                    Pipeline([raising(error)]).run(ctx=RunContext(recorder=recorder))
                self.assertIs(raised.exception, error)
                self.assertEqual(
                    [event.status for event in recorder.events[-2:]],
                    [Status.CANCELLED, Status.CANCELLED],
                )

    def test_exception_rendering_failure_preserves_original_error(self):
        class UnprintableError(Exception):
            def __str__(self):
                raise RuntimeError("message rendering failed")

        error = UnprintableError()
        with self.assertRaises(UnprintableError) as raised:
            Pipeline([raising(error)]).run(ctx=self.ctx)
        self.assertIs(raised.exception, error)
        self.assertEqual(self.recorder.events[-1].status, Status.FAILED)
        self.assertEqual(
            self.recorder.events[-1].error_message, "<exception message unavailable>"
        )

    def test_repeated_class_has_fresh_instances_and_unique_executions(self):
        instances = []

        class Capture(Add):
            def process(self, data, ctx):
                instances.append(self)
                return super().process(data, ctx)

        pipeline = Pipeline([Capture, Capture])
        self.assertEqual(pipeline.run(0, self.ctx), 2)
        self.assertEqual(pipeline.run(0, self.ctx), 2)
        self.assertEqual(len({id(stage) for stage in instances}), 4)
        self.assertTrue(all(stage.calls == 1 for stage in instances))
        starts = [
            event for event in self.recorder.events if event.status is Status.RUNNING
        ]
        self.assertEqual(len({event.execution_id for event in starts}), 6)

    def test_pipeline_snapshots_stage_sequence(self):
        stages = [Add]
        pipeline = Pipeline(stages)
        stages.append(Add)
        self.assertIsInstance(pipeline.stages, tuple)
        self.assertEqual(pipeline.run(0), 1)
        self.assertEqual(Pipeline(stage for stage in stages).run(0), 2)

    def test_stage_can_run_independently_with_monotonic_timing(self):
        with patch("expman.context.perf_counter", side_effect=[10.0, 12.5]):
            self.assertEqual(Add().run(1, self.ctx), 2)
        start, finish = self.recorder.events
        self.assertEqual(finish.duration_seconds, 2.5)
        self.assertIsNone(start.parent_id)
        self.assertIsNotNone(start.timestamp.tzinfo)

    def test_metrics_and_progress_belong_to_stage(self):
        Pipeline([Report]).run(ctx=self.ctx)
        stage = self.recorder.events[1]
        metric, progress = self.recorder.events[2:4]
        self.assertIsInstance(metric, MetricEvent)
        self.assertIsInstance(progress, ProgressEvent)
        self.assertEqual(metric.execution_id, stage.execution_id)
        self.assertEqual(progress.execution_id, stage.execution_id)
        self.assertEqual((metric.name, metric.value, metric.step), ("loss", 0.5, 2))
        self.assertEqual(
            (progress.completed, progress.total, progress.unit), (2, 4, "batch")
        )

    def test_nested_pipeline_preserves_parent_scopes(self):
        class Nested(Stage):
            def process(self, data, ctx):
                result = Pipeline([Add], name="inner").run(data, ctx)
                ctx.report_metric("result", result)
                return result

        self.assertEqual(Pipeline([Nested]).run(1, self.ctx), 2)
        outer, stage, inner, inner_stage = self.recorder.events[:4]
        self.assertEqual(stage.parent_id, outer.execution_id)
        self.assertEqual(inner.parent_id, stage.execution_id)
        self.assertEqual(inner_stage.parent_id, inner.execution_id)
        metric = next(
            event for event in self.recorder.events if isinstance(event, MetricEvent)
        )
        self.assertEqual(metric.execution_id, stage.execution_id)

    def test_recorder_failure_does_not_change_success(self):
        with self.assertLogs("expman.context", level="WARNING"):
            result = Pipeline([Report, Add]).run(
                2, RunContext(recorder=BrokenRecorder())
            )
        self.assertEqual(result, 3)

    def test_recorder_failure_does_not_mask_business_error(self):
        error = ValueError("business failure")
        with (
            self.assertLogs("expman.context", level="WARNING"),
            self.assertRaises(ValueError) as raised,
        ):
            Pipeline([raising(error)]).run(ctx=RunContext(recorder=BrokenRecorder()))
        self.assertIs(raised.exception, error)

    def test_default_run_contexts_are_independent(self):
        contexts = []

        class Capture(Stage):
            def process(self, data, ctx):
                contexts.append(ctx)
                return data

        pipeline = Pipeline([Capture])
        pipeline.run()
        pipeline.run()
        self.assertNotEqual(contexts[0].run_id, contexts[1].run_id)
        self.assertIsNot(contexts[0].recorder, contexts[1].recorder)

    def test_construction_validation(self):
        with self.assertRaises(TypeError):
            Stage()
        with self.assertRaises(TypeError):
            Pipeline([lambda data: data])
        with self.assertRaises(TypeError):
            Pipeline([Add()])
        for name in ("", " ", 42):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    Add(name=name)
                with self.assertRaises(ValueError):
                    Pipeline([], name=name)

    def test_metric_validation(self):
        for value in (float("nan"), float("inf"), -float("inf"), True, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.ctx.report_metric("loss", value)
        for step in (-1, True, 0.5):
            with self.subTest(step=step), self.assertRaises(ValueError):
                self.ctx.report_metric("loss", 1, step=step)
        with self.assertRaises(ValueError):
            self.ctx.report_metric(" ", 1)
        self.assertEqual(self.recorder.events, [])

    def test_progress_validation_and_unknown_total(self):
        invalid = [
            {"completed": -1},
            {"completed": True},
            {"completed": 0.5},
            {"completed": 1, "total": 0},
            {"completed": 0, "total": -1},
            {"completed": 0, "total": False},
            {"completed": 0, "unit": " "},
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.ctx.report_progress(**arguments)
        self.ctx.report_progress(0, total=0)
        self.ctx.report_progress(3, unit="epoch")
        self.assertIsNone(self.recorder.events[-1].total)


if __name__ == "__main__":
    unittest.main()
