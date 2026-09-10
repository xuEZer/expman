import io
import tempfile
import threading
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from expman import Batch, Pipeline, Stage, TimeEstimate
from expman.storage import read_record


class BatchTimingTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_resume_accumulates_wall_time_without_downtime(self):
        clock = [0.0]

        class Work(Stage):
            def process(self, data, ctx):
                clock[0] += 120
                if ctx.attempt == 1:
                    raise KeyboardInterrupt
                return 1

        with patch("expman.batch.monotonic", side_effect=lambda: clock[0]):
            pipeline = Pipeline([Work])
            batch = Batch(pipeline, {}, output_dir=self.root / "batch")
            with self.assertRaises(KeyboardInterrupt):
                batch.run(progress=False)
            self.assertEqual(batch.elapsed_seconds, 120)
            clock[0] += 10000
            resumed = Batch.resume(pipeline, batch.output_dir)
            self.assertEqual(resumed.elapsed_seconds, 120)
            output = io.StringIO()
            with redirect_stderr(output):
                resumed.run()
            self.assertIn("已运行 00:00:02", output.getvalue().splitlines()[0])
            self.assertEqual(resumed.elapsed_seconds, 240)
            again = Batch.resume(pipeline, batch.output_dir)
            self.assertEqual(again.elapsed_seconds, 240)

    def test_gpu_resume_uses_saved_interval_until_fresh_estimate(self):
        pipeline = Pipeline([])
        batch = Batch(pipeline, {"device": [0]}, output_dir=self.root / "batch")
        previous = TimeEstimate(100, 200, 0.8, 3, 1)
        batch._last_estimate = previous
        batch._elapsed_seconds = 60
        batch._save_timing()
        resumed = Batch.resume(pipeline, batch.output_dir)
        self.assertEqual(resumed.estimate(), previous)
        unknown = resumed.estimate(coverage=0.9)
        self.assertIsNone(unknown.lower_seconds)
        fresh = TimeEstimate(50, 90, 0.8, 3, 1)
        with patch("expman.parallel_estimation.estimate_parallel", return_value=fresh):
            self.assertEqual(resumed.estimate(), fresh)
        resumed._save_timing()
        self.assertEqual(Batch.resume(pipeline, batch.output_dir).estimate(), fresh)

    def test_periodic_timing_saves_without_cli(self):
        saved = threading.Event()

        class Work(Stage):
            def process(self, data, ctx):
                if not saved.wait(3):
                    raise RuntimeError("periodic save missing")
                return 1

        batch = Batch(Pipeline([Work]), {}, output_dir=self.root / "batch")
        original = batch._save_timing

        def persist():
            original()
            saved.set()

        with (
            patch("expman.batch.TIMING_SAVE_INTERVAL", 0.01),
            patch.object(batch, "_save_timing", side_effect=persist),
        ):
            result = batch.run(progress=False)[0]
        self.assertEqual(result.output, 1)
        record = read_record(batch.output_dir / "timing.pkl", batch._serializer)
        self.assertGreater(record["elapsed_seconds"], 0)
        self.assertEqual(record["estimate"]["upper_seconds"], 0)
        self.assertFalse(any(t.name == "expman-timing" for t in threading.enumerate()))

    def test_legacy_batch_without_timing_still_resumes(self):
        pipeline = Pipeline([])
        batch = Batch(pipeline, {}, output_dir=self.root / "batch")
        resumed = Batch.resume(pipeline, batch.output_dir)
        self.assertEqual(resumed.elapsed_seconds, 0)
        self.assertIsNone(resumed.estimate().lower_seconds)
