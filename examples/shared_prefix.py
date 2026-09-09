"""Reuse preparation when only the downstream model changes."""

from pathlib import Path
from tempfile import TemporaryDirectory

from expman import Batch, Pipeline, Stage


class Prepare(Stage):
    def process(self, data, ctx):
        print("Preparing data once")
        ctx.state["size"] = ctx.cfg["size"]
        return list(range(ctx.state["size"]))


class Predict(Stage):
    def process(self, data, ctx):
        return [value * ctx.cfg["scale"] for value in data]


def main():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        config = root / "experiment.yaml"
        config.write_text("size: 3\nscale: !choice [2, 3]\n")
        results = Batch(
            Pipeline([Prepare, Predict]), config, output_dir=root / "batch"
        ).run(progress=False)
        assert [result.output for result in results] == [[0, 2, 4], [0, 3, 6]]
        print([result.output for result in results])


if __name__ == "__main__":
    main()
