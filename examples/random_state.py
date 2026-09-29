"""Checkpoint recovery reproduces the uninterrupted random sequence."""

import random

from expman import Batch, Pipeline, Stage


class Sample(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {"count": True}

    def init(self, ctx):
        self.samples: list[float] = []

    def loop(self, data, ctx, index, max_iter=Stage.cfg("count")):
        self.samples.append(random.random())
        if ctx.attempt == 1 and index == ctx.cfg["count"] - 2:
            random.random()  # Consumed after the checkpoint; replayed on resume.
            raise RuntimeError("example: retry from checkpoint")
        return self.samples


def main():
    batch = Batch(Pipeline([Sample]), {"device": [0], "seed": 42, "count": 4})
    result = batch.run()[0]
    expected = random.Random(42)
    assert result.output == [expected.random() for _ in range(4)]
    print(result.output)


if __name__ == "__main__":
    main()
