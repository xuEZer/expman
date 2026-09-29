"""Checkpoint demo; use --interrupt-once, then --resume <printed directory>."""

import argparse

from expman import Batch, Pipeline, Stage


class LoadValues(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {"values": True}

    def process(self, data, ctx):
        return list(ctx.cfg["values"])


class Accumulate(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {"interrupt_once": True, "max_iter": True}

    def init(self, ctx):
        self.total = 0

    def loop(self, data, ctx, index, max_iter=Stage.cfg("max_iter")):
        self.total += data[index]
        if ctx.cfg["interrupt_once"] and ctx.attempt == 1 and index == 1:
            raise KeyboardInterrupt()
        return self.total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume")
    parser.add_argument("--interrupt-once", action="store_true")
    args = parser.parse_args()
    pipeline = Pipeline([LoadValues, Accumulate])
    batch = (
        Batch.resume(pipeline, args.resume)
        if args.resume
        else Batch(
            pipeline,
            {
                "device": [0],
                "seed": 0,
                "values": [1, 2, 3],
                "max_iter": 3,
                "interrupt_once": args.interrupt_once,
            },
        )
    )
    print(f"Batch directory: {batch.output_dir}", flush=True)
    try:
        for result in batch.run():
            print(f"{result.run_id}: {result.status.value}, output={result.output}")
    except KeyboardInterrupt:
        print(
            f"Resume with: python examples/checkpoint_pipeline.py --resume {batch.output_dir}"
        )
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
