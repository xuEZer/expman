"""Run independent experiments on the GPU list in a YAML file.

This example uses a timed placeholder; replace Work.process with GPU business code.
"""

import argparse
import os
from pathlib import Path
from time import sleep

from expman import Batch, Pipeline, Stage


class Work(Stage):
    def process(self, data, ctx):
        for epoch in range(ctx.state.get("epoch", 0), ctx.cfg["epochs"]):
            sleep(0.1)
            ctx.log_metrics({"train": {"loss": 1 / (epoch + 1)}}, step=epoch)
            ctx.state["epoch"] = epoch + 1
            ctx.checkpoint.save(step=epoch)
        return {
            "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "epochs": ctx.state["epoch"],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cfg", type=Path, default=Path(__file__).with_name("multi_gpu.yaml")
    )
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    pipeline = Pipeline([Work])
    batch = (
        Batch.resume(pipeline, args.resume)
        if args.resume
        else Batch(pipeline, args.cfg)
    )
    print(f"Batch directory: {batch.output_dir}")
    batch.run()


if __name__ == "__main__":
    main()
