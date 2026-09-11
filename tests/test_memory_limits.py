import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from expman import Batch, Pipeline, Stage, Status, limits
from expman.devices import (
    CAPACITY_FACTOR,
    CAPACITY_FLOOR_KB,
    DeviceMemory,
    HostMemory,
)
from expman.scheduling import GpuScheduler
from expman.scheduling import PeakEstimator as _PeakEstimator
from expman.storage import PickleSerializer, read_record

ORIGINAL_FROM_BATCH = _PeakEstimator.from_batch

ALLOCATION_MB = 900
ESTIMATE_DEFAULT_KB = 64 * 1024
ORIGINAL_MEMORY_LIMIT = limits.memory_limit


class Allocate(Stage):
    """Touch a fixed amount of host memory, so a cgroup cap can stop it."""

    def process(self, data, ctx):
        block = bytearray(ALLOCATION_MB * 1024 * 1024)
        for offset in range(0, len(block), 4096):
            block[offset] = 1
        return len(block)


class FakeGpu:
    def __init__(self, devices):
        self.devices = devices

    def sample(self):
        return {
            device: DeviceMemory(f"GPU-test-{device}", 1000, 900)
            for device in self.devices
        }


def small_default_estimator(batch):
    """The real estimator, with a cold-start default small enough to trip early."""
    estimator = ORIGINAL_FROM_BATCH(batch)
    estimator.default_kb = ESTIMATE_DEFAULT_KB
    return estimator


def roomy_host(*_args):
    """A host with plenty of room, so only the cgroup limit is under test."""

    class Host:
        def sample(self):
            return HostMemory(64 * 1024**2, 32 * 1024**2, 0, 0)

    return Host()


class MemoryLimitTests(unittest.TestCase):
    def test_cap_has_a_floor_and_grows_with_the_estimate(self):
        self.assertEqual(limits.cap_kb(1.0), CAPACITY_FLOOR_KB)
        estimate = 4 * 1024 * 1024
        self.assertEqual(limits.cap_kb(estimate), estimate * CAPACITY_FACTOR)

    def test_memory_limit_falls_back_to_a_systemd_scope(self):
        with (
            patch("expman.limits._direct", return_value=None),
            patch("expman.limits.shutil.which", return_value="/usr/bin/systemd-run"),
            patch(
                "expman.limits.own_cgroup",
                return_value=Path("/sys/fs/cgroup/user.slice/my.scope"),
            ),
        ):
            limit = limits.memory_limit("expman-test-1", 1024 * 1024)
        self.assertEqual(limit.mechanism, "systemd")
        self.assertIn("--scope", limit.command)
        self.assertIn("--quiet", limit.command)
        self.assertIn(f"--property=MemoryMax={int(limit.cap_kb)}K", limit.command)
        self.assertIn("--property=MemorySwapMax=0", limit.command)
        self.assertEqual(
            limit.cgroup, Path("/sys/fs/cgroup/user.slice/expman-test-1.scope")
        )

    def test_memory_limit_reports_when_nothing_can_cap_it(self):
        with (
            patch("expman.limits._direct", return_value=None),
            patch("expman.limits.shutil.which", return_value=None),
        ):
            limit = limits.memory_limit("expman-test-2", 1024 * 1024)
        self.assertEqual(limit.mechanism, "none")
        self.assertEqual(limit.cap_kb, 0.0)
        self.assertFalse(limit.capped)
        self.assertIsNone(limit.cgroup)

    def test_usage_and_events_are_read_from_the_cgroup_directory(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "memory.current").write_text("10485760\n")
        (root / "memory.peak").write_text("20971520\n")
        (root / "memory.events").write_text(
            "low 0\nhigh 1\nmax 2\noom 0\noom_kill 1\noom_group_kill 1\n"
        )
        self.assertEqual(limits.usage(root), (10240.0, 20480.0))
        self.assertEqual(limits.events(root)["max"], 2.0)
        self.assertTrue(limits.refused(limits.events(root)))
        (root / "memory.events").write_text("max 0\noom 0\noom_kill 0\n")
        self.assertFalse(limits.refused(limits.events(root)))

    def test_a_peak_close_to_the_limit_counts_as_hitting_it(self):
        self.assertFalse(limits.near_cap(1000.0, 0.0))
        self.assertFalse(limits.near_cap(1000.0, 2000.0))
        self.assertTrue(limits.near_cap(1900.0, 2000.0))

    def test_release_removes_a_directly_created_cgroup(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        created = Path(directory.name) / "expman-unit-1"
        created.mkdir()
        limit = limits.MemoryLimit((), created, 1024.0, "cgroup")
        self.assertTrue(limits.release(limit))
        self.assertFalse(created.exists())
        self.assertFalse(limits.release(limit))
        scope = limits.MemoryLimit((), created, 1024.0, "systemd")
        self.assertFalse(limits.release(scope))
        self.assertFalse(limits.release(None))

    def test_unreadable_cgroup_state_is_not_pressure(self):
        missing = Path(tempfile.gettempdir()) / "expman-missing-cgroup"
        self.assertIsNone(limits.usage(missing))
        self.assertEqual(limits.events(missing), {})
        self.assertFalse(limits.refused({}))
        self.assertEqual(limits.usage(None), None)

    def test_watch_reports_a_refused_charge_and_the_final_peak(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        cgroup = root / "cgroup"
        cgroup.mkdir()
        (cgroup / "memory.current").write_text("4096\n")
        (cgroup / "memory.peak").write_text("8192\n")
        (cgroup / "memory.events").write_text("max 0\noom_kill 0\n")
        record = root / "memory_cap.pkl"
        stopped = threading.Event()

        def trip_the_limit():
            (cgroup / "memory.events").write_text("max 1\noom_kill 0\n")

        with patch("expman.limits.own_cgroup", return_value=cgroup):
            timer = threading.Timer(0.05, trip_the_limit)
            timer.start()
            limits.watch(record, 0.01, stopped)
            timer.cancel()
        report = read_record(record, PickleSerializer())
        self.assertTrue(limits.refused(report["events"]))
        self.assertEqual(report["usage"], (4.0, 8.0))


@unittest.skipUnless(
    ORIGINAL_MEMORY_LIMIT("expman-probe", 1024 * 1024).capped,
    "no cgroup or systemd scope available to cap a worker",
)
class CappedWorkerTests(unittest.TestCase):
    """End to end: the cap stops an attempt, the estimate rises, the retry fits."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for target, value in [
            ("expman.devices.NvidiaMemory", FakeGpu),
            ("expman.devices.MeminfoMonitor", roomy_host),
            ("expman.devices.POLL_INTERVAL", 0.05),
            ("expman.scheduling.PeakEstimator.from_batch", small_default_estimator),
        ]:
            started = patch(target, value)
            started.start()
            self.addCleanup(started.stop)

    @staticmethod
    def cgroup_root():
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from expman import limits as limits_module

        own = limits_module.own_cgroup()
        return own.parent

    def test_capped_attempt_is_retried_with_a_raised_estimate(self):
        before = set(self.cgroup_root().glob("expman-*"))
        path = self.root / "experiments.yaml"
        path.write_text(
            yaml.safe_dump({"device": [0], "markers": str(self.root), "delay": 0})
            + "item: !choice [0]\n"
        )
        batch = Batch(Pipeline([Allocate]), path, output_dir=self.root / "batch")
        result = batch.run(progress=False)[0]
        self.assertEqual(result.attempts[0].status, Status.CANCELLED)
        self.assertEqual(result.attempts[0].error_type, "MemoryLimit")
        self.assertEqual(result.status, Status.SUCCEEDED)
        capped = [item for item in batch._gpu_history if item.get("capped")]
        self.assertTrue(capped)
        self.assertGreater(capped[0]["peak_kb"], CAPACITY_FLOOR_KB * 0.5)
        # The raised floor covers more than the peak that tripped the cap.
        estimate = GpuScheduler(batch).peaks.estimate_kb(result.run_id)
        self.assertGreater(estimate, capped[0]["peak_kb"])
        if ORIGINAL_MEMORY_LIMIT("probe", 1.0).mechanism == "cgroup":
            # Each attempt's cgroup is removed again, so a large Batch does not
            # leave one directory per attempt behind.
            self.assertEqual(
                sorted(self.cgroup_root().glob("expman-*"))[len(before) :], []
            )


if __name__ == "__main__":
    unittest.main()
