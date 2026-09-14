import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from expman import limits


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
