from unittest.mock import patch

import pytest

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


@pytest.fixture
def recorder():
    return InMemoryRecorder()


@pytest.fixture
def ctx(recorder):
    return RunContext(recorder=recorder)


def test_order_and_execution_hierarchy(recorder, ctx):
    class AddThree(Add):
        def __init__(self):
            super().__init__(3)

    class Double(Stage):
        def process(self, data, ctx):
            return data * 2

    result = Pipeline([AddThree, Double]).run(2, ctx)
    assert result == 10
    events = recorder.events
    assert [event.status for event in events] == [
        Status.RUNNING,
        Status.RUNNING,
        Status.SUCCEEDED,
        Status.RUNNING,
        Status.SUCCEEDED,
        Status.SUCCEEDED,
    ]
    assert all(event.run_id == ctx.run_id for event in events)
    assert events[0].parent_id is None
    for index in (1, 2, 3, 4):
        assert events[index].parent_id == events[0].execution_id
    assert events[0].execution_id == events[-1].execution_id
    assert events[1].execution_id == events[2].execution_id
    assert ctx.execution_id is None


def test_empty_pipeline_preserves_input_identity(recorder):
    data = object()
    assert Pipeline([]).run(data, RunContext(recorder=recorder)) is data
    assert len(recorder.events) == 2
    assert Pipeline([]).run() is None


def test_arbitrary_outputs_including_none_are_forwarded():
    class Clear(Stage):
        def process(self, data, ctx):
            return None

    class Wrap(Stage):
        def process(self, data, ctx):
            return {"value": data}

    assert Pipeline([Clear, Wrap]).run(object()) == {"value": None}


def test_failure_stops_downstream_and_preserves_exception(recorder, ctx):
    error = ValueError("bad input")
    calls = []

    class Downstream(Stage):
        def process(self, data, ctx):
            calls.append(data)

    with pytest.raises(ValueError) as raised:
        Pipeline([raising(error), Downstream]).run(1, ctx)
    assert raised.value is error
    assert calls == []
    finished = recorder.events[-2:]
    assert [event.status for event in finished] == [Status.FAILED] * 2
    for event in finished:
        assert event.error_type == "ValueError"
        assert event.error_message == "bad input"
        assert event.duration_seconds >= 0


@pytest.mark.parametrize(
    "error",
    [KeyboardInterrupt(), SystemExit(2)],
    ids=["KeyboardInterrupt", "SystemExit"],
)
def test_cancellation_is_recorded_and_propagated(error):
    recorder = InMemoryRecorder()
    with pytest.raises(type(error)) as raised:
        Pipeline([raising(error)]).run(ctx=RunContext(recorder=recorder))
    assert raised.value is error
    assert [event.status for event in recorder.events[-2:]] == [
        Status.CANCELLED,
        Status.CANCELLED,
    ]


def test_exception_rendering_failure_preserves_original_error(recorder, ctx):
    class UnprintableError(Exception):
        def __str__(self):
            raise RuntimeError("message rendering failed")

    error = UnprintableError()
    with pytest.raises(UnprintableError) as raised:
        Pipeline([raising(error)]).run(ctx=ctx)
    assert raised.value is error
    assert recorder.events[-1].status is Status.FAILED
    assert recorder.events[-1].error_message == "<exception message unavailable>"


def test_repeated_class_has_fresh_instances_and_unique_executions(recorder, ctx):
    instances = []

    class Capture(Add):
        def process(self, data, ctx):
            instances.append(self)
            return super().process(data, ctx)

    pipeline = Pipeline([Capture, Capture])
    assert pipeline.run(0, ctx) == 2
    assert pipeline.run(0, ctx) == 2
    assert len({id(stage) for stage in instances}) == 4
    assert all(stage.calls == 1 for stage in instances)
    starts = [event for event in recorder.events if event.status is Status.RUNNING]
    assert len({event.execution_id for event in starts}) == 6


def test_pipeline_snapshots_stage_sequence():
    stages = [Add]
    pipeline = Pipeline(stages)
    stages.append(Add)
    assert isinstance(pipeline.stages, tuple)
    assert pipeline.run(0) == 1
    assert Pipeline(stage for stage in stages).run(0) == 2


def test_stage_can_run_independently_with_monotonic_timing(recorder, ctx):
    with patch("expman.context.perf_counter", side_effect=[10.0, 12.5]):
        assert Add().run(1, ctx) == 2
    start, finish = recorder.events
    assert finish.duration_seconds == 2.5
    assert start.parent_id is None
    assert start.timestamp.tzinfo is not None


def test_metrics_and_progress_belong_to_stage(recorder, ctx):
    Pipeline([Report]).run(ctx=ctx)
    stage = recorder.events[1]
    metric, progress = recorder.events[2:4]
    assert isinstance(metric, MetricEvent)
    assert isinstance(progress, ProgressEvent)
    assert metric.execution_id == stage.execution_id
    assert progress.execution_id == stage.execution_id
    assert (metric.name, metric.value, metric.step) == ("loss", 0.5, 2)
    assert (progress.completed, progress.total, progress.unit) == (2, 4, "batch")


def test_nested_pipeline_preserves_parent_scopes(recorder, ctx):
    class Nested(Stage):
        def process(self, data, ctx):
            result = Pipeline([Add], name="inner").run(data, ctx)
            ctx.report_metric("result", result)
            return result

    assert Pipeline([Nested]).run(1, ctx) == 2
    outer, stage, inner, inner_stage = recorder.events[:4]
    assert stage.parent_id == outer.execution_id
    assert inner.parent_id == stage.execution_id
    assert inner_stage.parent_id == inner.execution_id
    metric = next(event for event in recorder.events if isinstance(event, MetricEvent))
    assert metric.execution_id == stage.execution_id


def test_recorder_failure_does_not_change_success(caplog):
    with caplog.at_level("WARNING", logger="expman.context"):
        result = Pipeline([Report, Add]).run(2, RunContext(recorder=BrokenRecorder()))
    assert result == 3
    assert caplog.records


def test_recorder_failure_does_not_mask_business_error(caplog):
    error = ValueError("business failure")
    with (
        caplog.at_level("WARNING", logger="expman.context"),
        pytest.raises(ValueError) as raised,
    ):
        Pipeline([raising(error)]).run(ctx=RunContext(recorder=BrokenRecorder()))
    assert raised.value is error
    assert caplog.records


def test_default_run_contexts_are_independent():
    contexts = []

    class Capture(Stage):
        def process(self, data, ctx):
            contexts.append(ctx)
            return data

    pipeline = Pipeline([Capture])
    pipeline.run()
    pipeline.run()
    assert contexts[0].run_id != contexts[1].run_id
    assert contexts[0].recorder is not contexts[1].recorder


def test_construction_validation():
    with pytest.raises(TypeError):
        Stage()
    with pytest.raises(TypeError):
        Pipeline([lambda data: data])
    with pytest.raises(TypeError):
        Pipeline([Add()])


@pytest.mark.parametrize("name", ["", " ", 42])
def test_invalid_names_are_rejected(name):
    with pytest.raises(ValueError):
        Add(name=name)
    with pytest.raises(ValueError):
        Pipeline([], name=name)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf"), True, "1"]
)
def test_invalid_metric_values_are_rejected(ctx, value):
    with pytest.raises(ValueError):
        ctx.report_metric("loss", value)


@pytest.mark.parametrize("step", [-1, True, 0.5])
def test_invalid_metric_steps_are_rejected(ctx, step):
    with pytest.raises(ValueError):
        ctx.report_metric("loss", 1, step=step)


def test_blank_metric_name_is_rejected(recorder, ctx):
    with pytest.raises(ValueError):
        ctx.report_metric(" ", 1)
    assert recorder.events == []


@pytest.mark.parametrize(
    "arguments",
    [
        {"completed": -1},
        {"completed": True},
        {"completed": 0.5},
        {"completed": 1, "total": 0},
        {"completed": 0, "total": -1},
        {"completed": 0, "total": False},
        {"completed": 0, "unit": " "},
    ],
)
def test_invalid_progress_arguments_are_rejected(ctx, arguments):
    with pytest.raises(ValueError):
        ctx.report_progress(**arguments)


def test_progress_accepts_zero_total_and_unknown_total(recorder, ctx):
    ctx.report_progress(0, total=0)
    ctx.report_progress(3, unit="epoch")
    assert recorder.events[-1].total is None
