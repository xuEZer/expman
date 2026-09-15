import unittest

from expman._memory_model import LocalQuantileEstimator, PeakCeiling


class LocalQuantileEstimatorTests(unittest.TestCase):
    def test_estimate_is_bounded_by_the_finite_local_samples(self):
        vectors = [[1.0, value] for value in range(6)]
        indices = {f"run-{index}": index for index in range(6)}
        estimator = LocalQuantileEstimator(vectors, indices, default=1024.0)
        for index, peak in enumerate((300.0, 340.0, 375.0, 410.0, 450.0)):
            estimator.record(f"run-{index}", peak)

        estimate = estimator.estimate("run-5")

        self.assertGreaterEqual(estimate, 300.0)
        self.assertLessEqual(estimate, 450.0)

    def test_capped_sample_has_one_finite_retry_bump(self):
        vectors = [[1.0], [1.0]]
        indices = {"retry": 0, "other": 1}
        estimator = LocalQuantileEstimator(vectors, indices, default=1024.0)
        estimator.record("retry", 800.0, capped=True)

        self.assertEqual(estimator.estimate("retry"), 1200.0)
        self.assertEqual(estimator.estimate("other"), 1200.0)

    def test_zero_gpu_telemetry_is_a_real_sample_when_requested(self):
        vectors = [[1.0], [1.0]]
        indices = {"completed": 0, "pending": 1}
        estimator = LocalQuantileEstimator(
            vectors, indices, default=1024.0, include_zero=True
        )
        estimator.record("completed", 0.0)

        self.assertEqual(estimator.estimate("pending"), 0.0)


class PeakCeilingTests(unittest.TestCase):
    def test_rebuild_preserves_capped_history_bump(self):
        ceiling = PeakCeiling.from_history(
            [
                {"peak_kb": 500.0, "capped": False},
                {"peak_kb": 800.0, "capped": True},
                {"peak_kb": float("inf"), "capped": False},
            ]
        )

        self.assertEqual(ceiling.ceiling_kb, 1200.0)
