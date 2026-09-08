"""Checkpoint demo; use --interrupt-once, then --resume <printed directory>."""

import argparse

from expman import Batch, Pipeline, Stage


class LoadValues(Stage):
    def process(self, data, ctx):
        ctx.state["total"] = 0
        return list(ctx.cfg["values"])


class Accumulate(Stage):
    def process(self, data, ctx):
        for index in range(ctx.state.get("next", 0), len(data)):
            ctx.state["total"] += data[index]
            ctx.state["next"] = index + 1
            ctx.checkpoint.save(step=index + 1)
            if ctx.cfg["interrupt_once"] and ctx.attempt == 1:
                raise KeyboardInterrupt()
        return ctx.state["total"]


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
            pipeline, {"values": [1, 2, 3], "interrupt_once": args.interrupt_once}
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
