"""Display experiment-level time intervals while a sequential batch runs."""

from pathlib import Path
from time import sleep

from expman import Batch, Pipeline, Stage


class Work(Stage):
    def process(self, data, ctx):
        sleep(0.1 * ctx.cfg["size"])
        score = 1 / ctx.cfg["size"]
        ctx.log_metrics({"score": score})
        return score


def main():
    batch = Batch(
        Pipeline([Work]),
        cfg=Path(__file__).with_name("timed_experiments.yaml"),
        estimate_coverage=0.8,
    )
    batch.run(refresh_interval=0.25)


if __name__ == "__main__":
    main()
