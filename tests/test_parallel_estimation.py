import tempfile
import unittest
from pathlib import Path

from expman import Batch, Pipeline, Stage, Status
from expman.scheduling import GpuScheduler


class First(Stage):
    def process(self, data, ctx):
        return data


class Second(Stage):
    def process(self, data, ctx):
        return ctx.cfg["item"]


class ParallelEstimationTests(unittest.TestCase):
    def test_eta_counts_only_unmaterialized_stage_parameter_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = root / "cfg.yaml"
            cfg.write_text("device: [0]\nseed: 0\nitem: !choice [0, 1, 2]\n")
            batch = Batch(Pipeline([First, Second]), cfg, output_dir=root / "batch")
            first = batch.experiments[0]
            batch._queue.remove(first.run_id)
            batch._stage_history.extend(
                [
                    {
                        "run_id": first.run_id,
                        "stage": 0,
                        "status": Status.SUCCEEDED.value,
                        "reused": False,
                        "stage_duration_seconds": 10.0,
                        "dependencies": [],
                    },
                    {
                        "run_id": first.run_id,
                        "stage": 1,
                        "status": Status.SUCCEEDED.value,
                        "reused": False,
                        "stage_duration_seconds": 20.0,
                        "dependencies": [(("item",), "sample")],
                    },
                ]
            )
            scheduler = GpuScheduler(batch)

            estimate = batch.estimate()

            # Stage 0 has one config-independent cache group, already completed.
            # Stage 1 still needs one representative for item=1 and item=2.
            self.assertEqual(estimate.lower_seconds, 40.0)
            self.assertEqual(estimate.upper_seconds, 40.0)
            self.assertEqual(estimate.completed_samples, 2)
            self.assertEqual(estimate.remaining_experiments, 2)
            self.assertIs(batch._scheduler, scheduler)


if __name__ == "__main__":
    unittest.main()
