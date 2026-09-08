import math
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from expman import Batch, Pipeline, Stage, Status, StorageError, TimeEstimate
from expman._time_model import DurationModel, features
from expman.storage import read_record, write_record


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class EstimationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.clock = Clock()
        timer = patch("expman.experiment.perf_counter", self.clock)
        timer.start()
        self.addCleanup(timer.stop)

    def batch(self, stages, text="job: !choice [A, B, C]", **kwargs):
        config = self.root / "experiments.yaml"
        config.write_text(text)
        return Batch(Pipeline(stages), config, output_dir=self.root / "batch", **kwargs)

    def test_format_interval_rounding_and_large_days(self):
        def display(low, high):
            return str(TimeEstimate(low, high, 0.8, 1, 1))

        self.assertEqual(display(None, None), "??:??:??～??:??:??")
        self.assertEqual(display(0, 0), "00:00:00～00:00:00")
        self.assertEqual(display(59.99, 60.01), "00:00:00～00:00:02")
        self.assertEqual(display(86399, 86401), "00:23:59～01:00:01")
        self.assertEqual(display(100 * 86400, 100 * 86400), "100:00:00～100:00:00")
        self.assertEqual(display(60, math.inf), "00:00:01～??:??:??")

    def test_cold_start_and_first_completed_experiment(self):
        clock = self.clock

        class Work(Stage):
            def process(self, data, ctx):
                clock.value += 120

        batch = self.batch([Work])
        self.assertEqual(str(batch.estimate()), "??:??:??～??:??:??")
        batch.experiments[0].run()
        estimate = batch.estimate()
        self.assertEqual(estimate.completed_samples, 1)
        self.assertEqual(estimate.remaining_experiments, 2)
        self.assertLess(estimate.lower_seconds, estimate.upper_seconds)
        self.assertFalse(estimate.calibrated)
        self.assertEqual(batch.estimate(), estimate)
        state = random.getstate()
        batch.estimate()
        self.assertEqual(random.getstate(), state)

    def test_coverage_nesting_and_validation(self):
        clock = self.clock

        class Work(Stage):
            def process(self, data, ctx):
                clock.value += 120

        batch = self.batch([Work], estimate_coverage=0.9)
        self.assertEqual(batch.estimate().coverage, 0.9)
        batch.experiments[0].run()
        narrow, broad = batch.estimate(coverage=0.5), batch.estimate(coverage=0.95)
        self.assertLessEqual(broad.lower_seconds, narrow.lower_seconds)
        self.assertGreaterEqual(broad.upper_seconds, narrow.upper_seconds)
        for coverage in [0, 1, -0.1, True, "0.8", math.nan, math.inf]:
            with self.subTest(coverage=coverage), self.assertRaises(ValueError):
                batch.estimate(coverage=coverage)
        with self.assertRaises(ValueError):
            Batch(
                Pipeline([]), {}, output_dir=self.root / "invalid", estimate_coverage=0
            )
        self.assertFalse((self.root / "invalid").exists())

    def test_running_observation_changes_prediction_without_progress_calls(self):
        clock = self.clock
        estimates = []

        class Work(Stage):
            def process(self, data, ctx):
                if ctx.cfg["job"] == "A":
                    clock.value += 10
                else:
                    estimates.append(batch.estimate())
                    clock.value += 1000
                    estimates.append(batch.estimate())

        batch = self.batch([Work], "job: !choice [A, B]")
        batch.run()
        before, after = estimates
        self.assertEqual(after.completed_samples, 1)
        self.assertEqual(after.remaining_experiments, 1)
        self.assertGreater(after.upper_seconds, before.upper_seconds)
        self.assertGreaterEqual(after.lower_seconds, 0)
        self.assertEqual(str(batch.estimate()), "00:00:00～00:00:00")

    def test_schedule_covers_unmeasured_combinations(self):
        order = []
        clock = self.clock

        class Work(Stage):
            def process(self, data, ctx):
                order.append((ctx.cfg["model"], ctx.cfg["seed"]))
                clock.value += 60
                return dict(ctx.cfg)

        batch = self.batch([Work], "model: !choice [A, B]\nseed: !choice [1, 2, 3]\n")
        results = batch.run()
        self.assertEqual(order[0], ("A", 1))
        self.assertEqual(order[1][0], "B")
        self.assertEqual([r.output["model"] for r in results], ["A"] * 3 + ["B"] * 3)
        self.assertEqual(len(set(order)), 6)

    def test_cancellation_resume_uses_one_cumulative_observation(self):
        clock = self.clock
        starts = []

        class Work(Stage):
            def process(self, data, ctx):
                starts.append((ctx.cfg["job"], ctx.checkpoint.step))
                clock.value += 100 if ctx.cfg["job"] == "A" else 200
                if ctx.cfg["job"] == "B" and ctx.attempt == 1:
                    ctx.state["value"] = 1
                    ctx.checkpoint.save(step=1)
                    raise KeyboardInterrupt
                return ctx.cfg["job"]

        batch = self.batch([Work], estimate_coverage=0.9)
        with self.assertRaises(KeyboardInterrupt):
            batch.run()
        estimate = batch.estimate()
        resumed = Batch.resume(Pipeline([Work]), batch.output_dir)
        self.assertEqual(resumed.estimate(), estimate)
        self.assertEqual(resumed.estimate_coverage, 0.9)
        resumed.run()
        self.assertEqual(starts[2], ("B", 1))
        self.assertEqual(resumed.estimate().completed_samples, 3)
        self.assertEqual(resumed.results[1].duration_seconds, 400)
        self.assertEqual(str(resumed.estimate()), "00:00:00～00:00:00")

    def test_failed_attempts_do_not_become_success_samples(self):
        clock = self.clock
        during = []

        class Fail(Stage):
            def process(self, data, ctx):
                clock.value += 20
                during.append(batch.estimate())
                raise ValueError("failure")

        batch = self.batch([Fail])
        batch.run()
        self.assertTrue(all(e.completed_samples == 0 for e in during))
        self.assertTrue(all(e.upper_seconds is None for e in during))
        self.assertEqual(str(batch.estimate()), "00:00:00～00:00:00")
        self.assertTrue(all(r.status is Status.FAILED for r in batch.results))

    def test_manifest_defaults_and_environment_validation(self):
        batch = self.batch([])
        path = batch.output_dir / "batch.pkl"
        manifest = read_record(path, batch._serializer)
        manifest.pop("estimation")
        write_record(path, manifest, batch._serializer)
        resumed = Batch.resume(Pipeline([]), batch.output_dir)
        self.assertEqual(resumed.estimate().coverage, 0.8)
        manifest["estimation"] = {
            "version": 1,
            "execution": "parallel",
            "coverage": 0.8,
        }
        write_record(path, manifest, batch._serializer)
        with self.assertRaises(StorageError):
            Batch.resume(Pipeline([]), batch.output_dir)

    def test_features_handle_arbitrary_yaml_structure_and_bound_dimensions(self):
        vectors = features(
            [
                {"nested": {"a/b": [1, None], "x": 10**400}, "name": "A"},
                {"nested": {"a": {"b": 2}, "x": -(10**400)}, "name": "B"},
            ]
        )
        self.assertNotEqual(vectors[0], vectors[1])
        self.assertTrue(all(math.isfinite(v) for row in vectors for v in row))
        many = features([{"label": str(i)} for i in range(100)])
        self.assertLessEqual(len(many[0]), 49)
        self.assertEqual(many, features([{"label": str(i)} for i in range(100)]))

    def test_joint_draws_retain_common_model_uncertainty(self):
        model = DurationModel([[1.0]] * 3, [(0, 100)], [])
        one = sorted(model.draws([(1, 0)]))
        two = sorted(model.draws([(1, 0), (2, 0)]))
        # Uncertainty does not disappear just because more tasks share the model.
        self.assertGreater(two[-77] - two[76], one[-77] - one[76])

    def test_synthetic_lognormal_coverage_sanity(self):
        rng = random.Random(17)
        covered = 0
        repetitions = 40
        for _ in range(repetitions):
            data = [math.exp(rng.gauss(math.log(100), 0.6)) for _ in range(24)]
            model = DurationModel([[1.0]] * 24, list(enumerate(data[:20])), [])
            draws = sorted(model.draws([(i, 0) for i in range(20, 24)]))
            covered += draws[76] <= sum(data[20:]) <= draws[-77]
        self.assertGreaterEqual(covered, repetitions * 0.6)
        self.assertLess(covered, repetitions)

    def test_estimate_can_be_polled_from_another_thread(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event

        ready, release = Event(), Event()
        clock = self.clock

        class Work(Stage):
            def process(self, data, ctx):
                clock.value += 100
                if ctx.cfg["job"] == "B":
                    ready.set()
                    if not release.wait(5):
                        raise RuntimeError("test did not release worker")

        batch = self.batch([Work], "job: !choice [A, B]")
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(batch.run)
            try:
                self.assertTrue(ready.wait(5))
                estimate = batch.estimate()
                self.assertEqual(estimate.completed_samples, 1)
                self.assertEqual(estimate.remaining_experiments, 1)
                self.assertIsNotNone(estimate.upper_seconds)
            finally:
                release.set()
            self.assertTrue(all(r.status is Status.SUCCEEDED for r in future.result(5)))
        self.assertEqual(str(batch.estimate()), "00:00:00～00:00:00")

    def test_hard_exit_resume_does_not_train_on_unknown_duration(self):
        import subprocess
        import sys

        script = self.root / "driver.py"
        script.write_text("""
import os
import sys
from pathlib import Path
from expman import Batch, Pipeline, Stage
root = Path(sys.argv[1])
class Work(Stage):
    def process(self, data, ctx):
        if ctx.cfg['job'] == 'B' and ctx.attempt == 1:
            ctx.checkpoint.save(step=1)
            os._exit(23)
        return ctx.cfg['job']
pipeline = Pipeline([Work])
if sys.argv[2] == 'new':
    cfg = root / 'config.yaml'
    cfg.write_text('job: !choice [A, B, C]\\n')
    Batch(pipeline, cfg, output_dir=root / 'batch').run()
else:
    batch = Batch.resume(pipeline, root / 'batch')
    assert batch.estimate().completed_samples == 1
    batch.run()
    assert batch.estimate().completed_samples == 2
    assert batch.results[1].attempts[0].error_type == 'InterruptedRun'
""")
        first = subprocess.run(
            [sys.executable, str(script), str(self.root), "new"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(first.returncode, 23, first.stderr)
        second = subprocess.run(
            [sys.executable, str(script), str(self.root), "resume"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(second.returncode, 0, second.stderr)

    def test_configuration_features_learn_different_experiment_costs(self):
        configs = [{"model": "small"}] * 9 + [{"model": "large"}] * 9
        observed = [(i, 100 + (i % 3 - 1) * 5) for i in range(8)]
        observed += [(i, 400 + (i % 3 - 1) * 20) for i in range(9, 17)]
        model = DurationModel(features(configs), observed, [])
        small = sorted(model.draws([(8, 0)]))[384]
        large = sorted(model.draws([(17, 0)]))[384]
        self.assertGreater(large / small, 2.5)
        self.assertLess(large / small, 5)
