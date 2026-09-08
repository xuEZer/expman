import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from expman import Batch, Pipeline, Status
from expman.experiment import AttemptResult
from expman.parallel_estimation import estimate_parallel


class ParallelEstimationTests(unittest.TestCase):
    def make_batch(self, root, devices):
        cfg = root / "cfg.yaml"
        cfg.write_text(f"device: {devices}\nitem: !choice [0, 1, 2]\n")
        batch = Batch(Pipeline([]), cfg, output_dir=root / "batch")
        first = batch.experiments[0]
        first._attempts.append(AttemptResult(first.run_id, 1, Status.SUCCEEDED, 10))
        batch._gpu_history.append(
            {
                "run_id": first.run_id,
                "attempt": 1,
                "device": 0,
                "uuid": "GPU-0",
                "concurrency": 1,
                "duration": 10,
            }
        )
        batch._queue.remove(first.run_id)
        batch._gpu_memory = dict.fromkeys(devices, 0.9)
        return batch

    def test_makespan_uses_individual_active_predictions(self):
        for devices, expected in (([0], 200), ([0, 1], 100)):
            with (
                self.subTest(devices=devices),
                tempfile.TemporaryDirectory() as temporary,
            ):
                batch = self.make_batch(Path(temporary), devices)
                second, third = batch.experiments[1:]
                batch._active_gpu[second.run_id] = 0
                batch._queue.remove(second.run_id)
                if len(devices) == 2:
                    batch._active_gpu[third.run_id] = 1
                    batch._queue.remove(third.run_id)
                with patch(
                    "expman.parallel_estimation.DurationModel.draws",
                    return_value=iter([[100, 100]]),
                ):
                    estimate = estimate_parallel(batch, 0.8)
                self.assertEqual(estimate.lower_seconds, expected)
                self.assertEqual(estimate.upper_seconds, expected)
                self.assertFalse(estimate.calibrated)

    def test_no_available_card_leaves_wait_time_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            batch = self.make_batch(Path(temporary), [0])
            batch._gpu_memory = {0: 0.05}
            self.assertIsNone(batch.estimate().upper_seconds)

    def test_observed_contention_is_used_as_a_model_feature(self):
        with tempfile.TemporaryDirectory() as temporary:
            batch = self.make_batch(Path(temporary), [0, 1])
            second = batch.experiments[1]
            batch._active_gpu[second.run_id] = 1
            batch._queue.remove(second.run_id)
            estimate = batch.estimate()
            self.assertEqual(estimate.completed_samples, 1)
            self.assertEqual(estimate.remaining_experiments, 2)
            self.assertGreaterEqual(estimate.upper_seconds, estimate.lower_seconds)


if __name__ == "__main__":
    unittest.main()
