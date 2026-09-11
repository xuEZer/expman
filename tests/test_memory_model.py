import tempfile
import unittest
from pathlib import Path

from expman import Batch, Pipeline, Stage
from expman._memory_model import PeakEstimator
from expman._time_model import features


def estimator(configs, history=()):
    vectors = features(configs)
    indices = {f"run-{index}": index for index in range(len(configs))}
    model = PeakEstimator(vectors, indices)
    for entry in history:
        model.record(entry)
    return model


class PeakEstimatorTests(unittest.TestCase):
    def test_cold_start_uses_the_batch_default(self):
        model = estimator([{"scale": 1}, {"scale": 2}])
        self.assertEqual(model.estimate_kb("run-0"), model.default_kb)
        self.assertEqual(model.estimate_kb("unknown-run"), model.default_kb)
        self.assertEqual(model.ceiling_kb, 0.0)

    def test_estimate_follows_configuration_features(self):
        configs = [{"scale": 1}, {"scale": 2}, {"scale": 4}]
        model = estimator(
            configs,
            [
                {"run_id": "run-0", "peak_kb": 100_000},
                {"run_id": "run-2", "peak_kb": 400_000},
            ],
        )
        small = model.estimate_kb("run-0")
        large = model.estimate_kb("run-2")
        # The upper quantile never promises less than what was observed, and the
        # larger configuration is estimated above the smaller one.
        self.assertGreaterEqual(small, 100_000)
        self.assertGreaterEqual(large, 400_000)
        self.assertGreater(large, small)
        self.assertEqual(model.ceiling_kb, 400_000)

    def test_capped_attempt_raises_the_floor_above_its_peak(self):
        configs = [{"scale": 1}, {"scale": 2}]
        model = estimator(configs, [{"run_id": "run-0", "peak_kb": 300_000}])
        model.record({"run_id": "run-1", "peak_kb": 5_000_000, "capped": True})
        floor = 5_000_000 * model.margin
        self.assertEqual(model.floors["run-1"], floor)
        self.assertGreaterEqual(model.estimate_kb("run-1"), floor)
        # Further complete observations narrow the regression, but the floor of a
        # capped run survives: that attempt is retried with room to grow.
        model.record({"run_id": "run-0", "attempt": 2, "peak_kb": 320_000})
        self.assertGreaterEqual(model.estimate_kb("run-1"), floor)
        self.assertEqual(model.ceiling_kb, 5_000_000)

    def test_capped_attempt_is_a_lower_bound_for_the_regression(self):
        configs = [{"scale": 1}, {"scale": 2}]
        model = estimator(
            configs,
            [
                {"run_id": "run-0", "peak_kb": 200_000},
                {"run_id": "run-1", "peak_kb": 900_000, "capped": True},
            ],
        )
        # Only one complete observation: the run without a bump still gets the
        # default, while the capped run keeps its raised floor.
        self.assertGreaterEqual(model.estimate_kb("run-1"), 900_000 * model.margin)
        model.record({"run_id": "run-0", "attempt": 2, "peak_kb": 250_000})
        self.assertGreaterEqual(model.estimate_kb("run-0"), 250_000)

    def test_entries_without_a_peak_or_row_are_ignored(self):
        model = estimator([{"scale": 1}], [{"run_id": "run-0", "peak_kb": 0}])
        self.assertEqual(model.observed, [])
        model.record({"run_id": "missing", "peak_kb": 1000})
        self.assertEqual(model.observed, [])
        self.assertEqual(model.ceiling_kb, 0.0)

    def test_batch_estimator_restores_history(self):
        class Publish(Stage):
            def process(self, data, ctx):
                return None

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        batch = Batch(
            Pipeline([Publish]),
            {"device": [0], "scale": 1},
            output_dir=Path(directory.name) / "batch",
        )
        entry = {
            "run_id": batch.experiments[0].run_id,
            "attempt": 1,
            "device": 0,
            "uuid": "GPU-0",
            "concurrency": 1.0,
            "duration": 1.0,
            "peak_kb": 123_456,
        }
        batch._gpu_history.append(entry)
        model = PeakEstimator.from_batch(batch)
        self.assertEqual(model.ceiling_kb, 123_456)
        self.assertGreaterEqual(model.estimate_kb(entry["run_id"]), 123_456)


if __name__ == "__main__":
    unittest.main()
