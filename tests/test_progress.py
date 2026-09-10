import io
import math
import os
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from expman import Batch, Pipeline, Stage, Status


class Output(io.StringIO):
    def __init__(self, *, tty=False, on_active=None):
        super().__init__()
        self.tty = tty
        self.on_active = on_active

    def isatty(self):
        return self.tty

    def fileno(self):
        return 2

    def write(self, text):
        result = super().write(text)
        if self.on_active is not None and "当前" in text:
            self.on_active(text)
        return result


class ProgressTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def batch(self, stages, cfg=None):
        return Batch(
            Pipeline(stages), {} if cfg is None else cfg, output_dir=self.root / "batch"
        )

    def test_default_progress_uses_stderr_and_leaves_stdout_for_business(self):
        class Work(Stage):
            def process(self, data, ctx):
                print("business output")
                return 42

        stderr, stdout = Output(), io.StringIO()
        with redirect_stderr(stderr), redirect_stdout(stdout):
            result = self.batch([Work]).run()
        self.assertEqual(result[0].output, 42)
        self.assertEqual(stdout.getvalue(), "business output\n")
        lines = stderr.getvalue().splitlines()
        self.assertIn("运行中 0/1 (0%)", lines[0])
        self.assertIn("??:??:??～??:??:??", lines[0])
        self.assertIn("已结束 1/1 (100%)", lines[-1])
        self.assertIn("成功 1 失败 0", lines[-1])
        self.assertIn("00:00:00～00:00:00", lines[-1])
        self.assertNotIn("\x1b", stderr.getvalue())
        self.assertNotIn("\r", stderr.getvalue())

    def test_elapsed_tracks_current_run_wall_time_in_days_hours_minutes(self):
        clock = [100.0]

        class Work(Stage):
            def process(self, data, ctx):
                clock[0] += 2 * 86400 + 3 * 3600 + 4 * 60 + 59

        output = Output()
        with (
            redirect_stderr(output),
            patch("expman.batch.monotonic", side_effect=lambda: clock[0]),
        ):
            self.batch([Work]).run()
        lines = output.getvalue().splitlines()
        self.assertIn("已运行 00:00:00", lines[0])
        self.assertIn("已运行 02:03:04", lines[-1])

    def test_tty_refreshes_while_process_is_still_running(self):
        observed = threading.Event()
        in_process = []

        class Work(Stage):
            def process(self, data, ctx):
                in_process.append(True)
                if not observed.wait(3):
                    raise RuntimeError("no live display before process completed")
                in_process.pop()

        def active(text):
            if in_process:
                observed.set()

        stderr = Output(tty=True, on_active=active)
        with (
            redirect_stderr(stderr),
            patch(
                "expman.progress.os.get_terminal_size",
                return_value=os.terminal_size((120, 24)),
            ),
        ):
            results = self.batch([Work]).run(refresh_interval=0.01)
        self.assertEqual(results[0].status, Status.SUCCEEDED)
        self.assertTrue(observed.is_set())
        self.assertIn("当前 1/1", stderr.getvalue())
        self.assertIn("\r\x1b[2K", stderr.getvalue())
        self.assertEqual(stderr.getvalue().count("\n"), 1)
        self.assertFalse(
            any(t.name == "expman-progress" for t in threading.enumerate())
        )

    def test_pending_retry_is_not_counted_as_finished_or_terminal_failure(self):
        observed = threading.Event()
        frames = []
        config = self.root / "experiments.yaml"
        config.write_text("job: !choice [A, B]\n")

        class Work(Stage):
            def process(self, data, ctx):
                if ctx.cfg["job"] == "A" and ctx.attempt == 1:
                    raise RuntimeError("retry")
                if ctx.cfg["job"] == "B" and not observed.wait(3):
                    raise RuntimeError("no progress frame")

        def active(text):
            frames.append(text)
            observed.set()

        stderr = Output(on_active=active)
        with redirect_stderr(stderr):
            results = self.batch([Work], config).run(refresh_interval=0.01)
        self.assertTrue(all(r.status is Status.SUCCEEDED for r in results))
        self.assertIn("0/2 (0%)", frames[0])
        self.assertIn("成功 0 失败 0", frames[0])
        self.assertIn("已结束 2/2 (100%)", stderr.getvalue())

    def test_terminal_failures_count_towards_finished_progress(self):
        class Fail(Stage):
            def process(self, data, ctx):
                raise RuntimeError("always fails")

        stderr = Output()
        with redirect_stderr(stderr):
            result = self.batch([Fail]).run()[0]
        self.assertEqual(len(result.attempts), 2)
        self.assertIn("已结束 1/1 (100%) | 成功 0 失败 1", stderr.getvalue())

    def test_interrupt_finishes_line_and_stops_monitor(self):
        class Interrupt(Stage):
            def process(self, data, ctx):
                raise KeyboardInterrupt

        stderr = Output(tty=True)
        batch = self.batch([Interrupt])
        with redirect_stderr(stderr), self.assertRaises(KeyboardInterrupt):
            batch.run(refresh_interval=0.01)
        self.assertIn("已中断 0/1 (0%)", stderr.getvalue())
        self.assertTrue(stderr.getvalue().endswith("\n"))
        self.assertEqual(batch.results[0].status, Status.CANCELLED)
        self.assertFalse(
            any(t.name == "expman-progress" for t in threading.enumerate())
        )

    def test_progress_can_be_disabled(self):
        batch = self.batch([])
        stderr = Output()
        with redirect_stderr(stderr):
            batch.run(progress=False)
        self.assertTrue((batch.output_dir / "timing.pkl").exists())
        self.assertEqual(stderr.getvalue(), "")

    def test_broken_output_does_not_fail_experiment(self):
        class BrokenOutput(Output):
            def write(self, text):
                raise BrokenPipeError("closed")

        with redirect_stderr(BrokenOutput()):
            result = self.batch([]).run()[0]
        self.assertEqual(result.status, Status.SUCCEEDED)
        self.assertEqual(len(result.attempts), 1)

    def test_estimator_failure_shows_unknown_without_retrying_business(self):
        batch = self.batch([])
        stderr = Output()
        with (
            redirect_stderr(stderr),
            patch.object(
                batch, "_display_estimate", side_effect=ValueError("unavailable")
            ),
            self.assertLogs("expman.progress", level="WARNING") as logs,
        ):
            result = batch.run()[0]
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(result.status, Status.SUCCEEDED)
        self.assertIn("??:??:??～??:??:??", stderr.getvalue())

    def test_invalid_display_options_leave_batch_runnable(self):
        batch = self.batch([])
        for interval in [0, -1, True, "fast", math.inf, math.nan]:
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                batch.run(refresh_interval=interval)
        with self.assertRaises(TypeError):
            batch.run(progress="yes")
        self.assertEqual(batch.run(progress=False)[0].status, Status.SUCCEEDED)

    def test_narrow_terminal_uses_complete_lines(self):
        stderr = Output(tty=True)
        with (
            redirect_stderr(stderr),
            patch(
                "expman.progress.os.get_terminal_size",
                return_value=os.terminal_size((20, 24)),
            ),
        ):
            self.batch([]).run()
        self.assertNotIn("\x1b", stderr.getvalue())
        self.assertEqual(len(stderr.getvalue().splitlines()), 2)
