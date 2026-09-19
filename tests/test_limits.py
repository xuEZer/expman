import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from expman import _worker, limits


class CgroupDiscoveryTests(unittest.TestCase):
    def test_cgroup_for_pid_uses_the_unified_hierarchy_reported_by_proc(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc_file = root / "proc" / "123" / "cgroup"
            proc_file.parent.mkdir(parents=True)
            proc_file.write_text("0::/user.slice/user-1000.slice/app.slice/run.scope\n")
            with patch("expman.limits.Path") as path:
                path.side_effect = lambda value: (
                    root / "proc" if str(value) == "/proc" else Path(value)
                )
                with patch("expman.limits.CGROUP_ROOT", root / "cgroup"):
                    self.assertEqual(
                        limits.cgroup_for_pid(123),
                        root
                        / "cgroup"
                        / "user.slice/user-1000.slice/app.slice/run.scope",
                    )

    def test_systemd_scope_does_not_guess_a_cgroup_path(self):
        completed = object()
        with (
            patch("expman.limits.shutil.which", return_value="/usr/bin/systemd-run"),
            patch("expman.limits.subprocess.run", return_value=completed),
        ):
            limit = limits._scope("expman-test", 1024)

        self.assertIsNotNone(limit)
        self.assertIsNone(limit.cgroup)
        self.assertEqual(limit.mechanism, "systemd")


class ColdStartLimitTests(unittest.TestCase):
    def test_cold_start_runs_uncapped_and_creates_no_cgroup(self):
        # The short circuit has to precede the mechanisms: a zero limit written
        # through _direct would be memory.max = 0, which cgroup v2 reads as
        # "this group may use no memory at all".
        with (
            patch("expman.limits._direct") as direct,
            patch("expman.limits._scope") as scope,
        ):
            limit = limits.memory_limit("expman-test", 1024, cold_start=True)

        direct.assert_not_called()
        scope.assert_not_called()
        self.assertEqual(limit.command, ())
        self.assertIsNone(limit.cgroup)
        self.assertEqual(limit.cap_kb, 0.0)
        self.assertFalse(limit.capped)
        self.assertEqual(limit.mechanism, "cold-start")

    def test_a_stage_without_a_cold_start_is_capped_from_its_estimate(self):
        with (
            patch("expman.limits._direct", return_value=None) as direct,
            patch("expman.limits._scope", return_value=None),
        ):
            limit = limits.memory_limit("expman-test", 2048 * 1024)

        requested = direct.call_args.args[1]
        self.assertEqual(requested, limits.cap_kb(2048 * 1024))
        self.assertGreater(requested, 2048 * 1024)
        self.assertFalse(limit.capped)
        self.assertEqual(limit.mechanism, "none")


class UncappedWorkerReportingTests(unittest.TestCase):
    def test_an_attempt_without_its_own_cgroup_reports_its_own_high_water_mark(self):
        # The enclosing group also holds the scheduler and every attempt it has
        # started, so its counters would charge this attempt for all of them.
        with patch.dict(os.environ, {"EXPMAN_CGROUP": ""}):
            self.assertIsNone(_worker._resource_cgroup())
            with patch.object(_worker, "gpu_memory", return_value={}):
                snapshot = _worker._resource_snapshot(_worker._resource_cgroup())

        current, peak = snapshot["host_current_kb"], snapshot["host_peak_kb"]
        self.assertGreater(current, 0)
        self.assertGreaterEqual(peak, current)
        self.assertIsNone(snapshot["cgroup"])
        self.assertEqual(snapshot["memory_events"], {})

    def test_a_worker_reads_the_cgroup_the_scheduler_gave_it(self):
        with patch.dict(os.environ, {"EXPMAN_CGROUP": "/sys/fs/cgroup/expman-x"}):
            self.assertEqual(
                _worker._resource_cgroup(), Path("/sys/fs/cgroup/expman-x")
            )
