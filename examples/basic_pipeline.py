"""Run with: python examples/basic_pipeline.py (after installing expman)."""

from dataclasses import dataclass

from expman import Batch, Pipeline, RunContext, Stage


class LoadNumbers(Stage[None, list[float]]):
    @classmethod
    def config_dependencies(cls, cfg):
        return {}

    def process(self, data: None, ctx: RunContext) -> list[float]:
        return [2.0, 4.0, 6.0]


class Scale(Stage[list[float], list[float]]):
    @classmethod
    def config_dependencies(cls, cfg):
        return {"factor": True}

    def process(self, data: list[float], ctx: RunContext) -> list[float]:
        result = []
        for index, value in enumerate(data, start=1):
            result.append(value * ctx.cfg["factor"])
            ctx.report_progress(index, total=len(data), unit="item")
        return result


@dataclass(frozen=True)
class Summary:
    count: int
    mean: float


class Summarize(Stage[list[float], Summary]):
    @classmethod
    def config_dependencies(cls, cfg):
        return {}

    def process(self, data: list[float], ctx: RunContext) -> Summary:
        mean = sum(data) / len(data)
        ctx.report_metric("mean", mean)
        return Summary(count=len(data), mean=mean)


def main() -> None:
    pipeline = Pipeline([LoadNumbers, Scale, Summarize], name="numbers")
    batch = Batch(pipeline, {"device": [0], "seed": 0, "factor": 0.5})
    for result in batch.run():
        print(
            f"{result.run_id} {result.status.value} "
            f"attempts={len(result.attempts)} output={result.output}"
        )


if __name__ == "__main__":
    main()
