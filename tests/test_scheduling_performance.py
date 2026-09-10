import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from expman import Batch, Pipeline, TimeEstimate
from expman._time_model import DurationModel, _solve, dot
from expman.estimation import TimeEstimator


class SchedulingPerformanceTests(unittest.TestCase):
    def test_cached_selection_never_refits_and_explores_diverse_configs(self):
        experiments = [
            SimpleNamespace(
                run_id=str(index), cfg={"x": value}, _timing_snapshot=lambda: ((), 0)
            )
            for index, value in enumerate((0, 1, 100))
        ]
        estimator = TimeEstimator(experiments)
        with patch(
            "expman.estimation.DurationModel",
            side_effect=AssertionError("dispatch refitted"),
        ):
            self.assertEqual(
                estimator.choose_cached(["0", "1", "2"], remaining_ids=["0", "1", "2"]),
                "0",
            )
            self.assertEqual(
                estimator.choose_cached(["1", "2"], remaining_ids=["1", "2"]), "2"
            )

    def test_projected_priority_matches_original_covariance_formula(self):
        vectors = [[1, 0, 0], [1, 1, 0], [1, 0, 1]]
        model = DurationModel(vectors, [(0, 10), (1, 20)], [])
        import math

        gradient = [
            sum(math.exp(dot(x, model.mean)) * x[j] for x in vectors) for j in range(3)
        ]
        actual = model.priorities([0, 1, 2])
        for index, x in enumerate(vectors):
            covariance = _solve(model.lower, x)
            expected = dot(gradient, covariance) ** 2 / (1 + dot(x, covariance))
            self.assertAlmostEqual(actual[index] / expected, 1)

    def test_cli_estimate_reads_cache_without_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = Batch(Pipeline([]), {}, output_dir=Path(directory) / "batch")
            batch._last_estimate = TimeEstimate(10, 20, 0.8, 1, 1)
            with patch.object(
                batch, "_refresh_estimate", side_effect=AssertionError("CLI refitted")
            ):
                self.assertEqual(batch._display_estimate(), batch._last_estimate)
                batch._queue.clear()
                self.assertEqual(batch._display_estimate().upper_seconds, 0)
