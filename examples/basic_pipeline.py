"""Run with: python examples/basic_pipeline.py (after installing expman)."""

from dataclasses import dataclass

from expman import ExecutionEvent, InMemoryRecorder, Pipeline, RunContext, Stage, Status


class LoadNumbers(Stage[None, list[float]]):
    def process(self, data: None, ctx: RunContext) -> list[float]:
        return [2.0, 4.0, 6.0]


class Scale(Stage[list[float], list[float]]):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def process(self, data: list[float], ctx: RunContext) -> list[float]:
        result = []
        for index, value in enumerate(data, start=1):
            result.append(value * self.factor)
            ctx.report_progress(index, total=len(data), unit="item")
        return result


@dataclass(frozen=True)
class Summary:
    count: int
    mean: float


class Summarize(Stage[list[float], Summary]):
    def process(self, data: list[float], ctx: RunContext) -> Summary:
        mean = sum(data) / len(data)
        ctx.report_metric("mean", mean)
        return Summary(count=len(data), mean=mean)


def main() -> None:
    recorder = InMemoryRecorder()
    context = RunContext(recorder=recorder)
    pipeline = Pipeline([LoadNumbers(), Scale(0.5), Summarize()], name="numbers")
    print(pipeline.run(ctx=context))
    for event in recorder.events:
        if isinstance(event, ExecutionEvent) and event.status is Status.SUCCEEDED:
            print(f"{event.kind}: {event.name} {event.duration_seconds:.6f}s")


if __name__ == "__main__":
    main()
