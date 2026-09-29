"""Run independent experiments on the GPU list in a YAML file.

This example uses a timed placeholder; replace Work.loop with GPU business code.
"""

import argparse
import os
from pathlib import Path
from time import sleep

from expman import Batch, Pipeline, Stage


class Work(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {"epochs": True}

    def init(self, ctx):
        self.epoch = 0

    def loop(self, data, ctx, index, max_iter=Stage.cfg("epochs")):
        sleep(0.1)
        ctx.log_metrics({"train": {"loss": 1 / (index + 1)}}, step=index)
        self.epoch = index + 1
        return {
            "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "epochs": self.epoch,
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
