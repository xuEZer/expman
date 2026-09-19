"""Checkpoint recovery reproduces the uninterrupted random sequence."""

import random

from expman import Batch, Pipeline, Stage


class Sample(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {"count": True}

    def process(self, data, ctx):
        samples = ctx.state.setdefault("samples", [])
        while len(samples) < ctx.cfg["count"]:
            samples.append(random.random())
            ctx.checkpoint.save(step=len(samples))
            if ctx.attempt == 1 and len(samples) == 2:
                random.random()  # Work after the checkpoint will be replayed.
                raise RuntimeError("example: retry from checkpoint")
        return samples


def main():
    batch = Batch(Pipeline([Sample]), {"seed": 42, "count": 4})
    result = batch.run()[0]
    expected = random.Random(42)
    assert result.output == [expected.random() for _ in range(4)]
    print(result.output)


if __name__ == "__main__":
    main()
