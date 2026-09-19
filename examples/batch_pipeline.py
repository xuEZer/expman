"""Run with: python examples/batch_pipeline.py."""

from pathlib import Path

from expman import Batch, Pipeline, Stage


class DescribeModels(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {"models": True, "seed": True}

    def process(self, data, ctx):
        models = ctx.cfg["models"]
        return {
            "seed": ctx.cfg["seed"],
            "imputation": models["imputation"]["name"],
            "forecasting": models["forecasting"]["name"],
            "patch_len": models["forecasting"]["patch_len"],
        }


class ReportSelection(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {}

    def process(self, data, ctx):
        ctx.report_metric("patch_len", data["patch_len"])
        return data


def main() -> None:
    pipeline = Pipeline([DescribeModels, ReportSelection], name="model-selection")
    batch = Batch(pipeline, cfg=Path(__file__).with_name("experiment.yaml"))
    for result in batch.run():
        print(
            f"{result.run_id} {result.status.value} "
            f"attempts={len(result.attempts)} output={result.output}"
        )


if __name__ == "__main__":
    main()
