from unittest.mock import patch

import pytest

from expman.scheduling import _worker, limits


def memory_group(path, *, version=2, current=4096, peak=8192, cap=16384):
    path.mkdir(parents=True, exist_ok=True)
    files = (
        ("memory.current", "memory.peak", "memory.max")
        if version == 2
        else (
            "memory.usage_in_bytes",
            "memory.max_usage_in_bytes",
            "memory.limit_in_bytes",
        )
    )
    (path / files[0]).write_text(str(current))
    if peak is not None:
        (path / files[1]).write_text(str(peak))
    (path / files[2]).write_text(str(cap))
    return path


def proc_layout(monkeypatch, tmp_path, memberships, mounts, pid="self"):
    root = tmp_path / "proc"
    directory = root / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "cgroup").write_text(memberships)
    (directory / "mountinfo").write_text(mounts)
    monkeypatch.setattr(limits, "PROC_ROOT", root)


def mount_line(path, *, version=2, root="/"):
    escaped = str(path).replace(" ", r"\040")
    tail = "cgroup2 cgroup rw" if version == 2 else "cgroup cgroup rw,memory"
    return f"29 23 0:26 {root} {escaped} rw - {tail}\n"


def test_v2_resolution_uses_mountinfo_instead_of_fixed_root(monkeypatch, tmp_path):
    mount = tmp_path / "unified"
    actual = memory_group(mount / "user.slice/expman-test.scope")
    proc_layout(
        monkeypatch,
        tmp_path,
        "0::/user.slice/expman-test.scope\n",
        mount_line(mount),
        pid=123,
    )
    assert limits.cgroup_for_pid(123) == actual


def test_resolution_handles_mount_root_and_escaped_mountpoint(monkeypatch, tmp_path):
    mount = tmp_path / "memory mount"
    actual = memory_group(mount / "expman-test.scope")
    proc_layout(
        monkeypatch,
        tmp_path,
        "0::/tenant/expman-test.scope\n",
        mount_line(mount, root="/tenant"),
    )
    assert limits.own_cgroup() == actual


def test_hybrid_prefers_real_v1_memory_controller_and_rejects_shared_group(
    monkeypatch, tmp_path
):
    v1, unified = tmp_path / "memory", tmp_path / "unified"
    shared = memory_group(
        v1 / "user.slice/user@1070.service", version=1, cap=9223372036854771712
    )
    (unified / "user.slice/user@1070.service/expman-test.scope").mkdir(parents=True)
    proc_layout(
        monkeypatch,
        tmp_path,
        "3:memory:/user.slice/user@1070.service\n0::/user.slice/user@1070.service/expman-test.scope\n",
        mount_line(unified) + mount_line(v1, version=1),
    )
    monkeypatch.setenv("EXPMAN_CGROUP_UNIT", "expman-test")
    monkeypatch.delenv("EXPMAN_CGROUP", raising=False)
    monkeypatch.setenv("EXPMAN_HOST_LIMIT_KB", "16384")
    assert limits.own_cgroup() == shared
    assert _worker._resource_cgroup() is None
    with (
        patch.object(_worker, "_own_usage", return_value=(3993208.0, 3993208.0)),
    ):
        snapshot = _worker._resource_snapshot(_worker._resource_cgroup())
    assert snapshot["host_current_kb"] == 3993208.0
    assert snapshot["host_usage_source"] == "proc"
    assert snapshot["host_limit_verified"] is False
    assert snapshot["host_limit_kb"] is None
    assert snapshot["cgroup"] is None


def test_v1_dedicated_group_is_measured_and_limit_verified(monkeypatch, tmp_path):
    mount = tmp_path / "memory"
    group = memory_group(mount / "user.slice/expman-test.scope", version=1)
    proc_layout(
        monkeypatch,
        tmp_path,
        "3:memory:/user.slice/expman-test.scope\n",
        mount_line(mount, version=1),
    )
    monkeypatch.setenv("EXPMAN_CGROUP_UNIT", "expman-test")
    monkeypatch.setenv("EXPMAN_HOST_LIMIT_KB", "16")
    monkeypatch.delenv("EXPMAN_CGROUP", raising=False)
    assert _worker._resource_cgroup() == group
    snapshot = _worker._resource_snapshot(group)
    assert snapshot["host_current_kb"] == 4
    assert snapshot["host_peak_kb"] == 8
    assert snapshot["host_limit_kb"] == 16
    assert snapshot["host_limit_verified"] is True
    assert snapshot["host_usage_source"] == "cgroup"


def test_systemd_scope_does_not_claim_limit_before_verification():
    with (
        patch(
            "expman.scheduling.limits.shutil.which", return_value="/usr/bin/systemd-run"
        ),
        patch("expman.scheduling.limits.subprocess.run", return_value=object()),
        patch.object(limits, "own_cgroup", return_value=None),
    ):
        cap = limits._scope("expman-test", 1024)
    assert cap is not None
    assert cap.cgroup is None
    assert cap.mechanism == "systemd"
    assert cap.cap_kb == 1024
    assert not cap.verified
    assert not cap.capped


def test_v1_systemd_requests_legacy_memory_limit(tmp_path):
    group = memory_group(tmp_path / "service", version=1)
    with (
        patch(
            "expman.scheduling.limits.shutil.which", return_value="/usr/bin/systemd-run"
        ),
        patch("expman.scheduling.limits.subprocess.run", return_value=object()),
        patch.object(limits, "own_cgroup", return_value=group),
    ):
        cap = limits._scope("expman-test", 1024)
    assert "--property=MemoryLimit=1024K" in cap.command
    assert not cap.capped


def test_memory_limit_without_working_mechanism_is_uncapped():
    with (
        patch.object(limits, "_direct", return_value=None) as direct,
        patch.object(limits, "_scope", return_value=None),
    ):
        cap = limits.memory_limit("expman-test", 2048 * 1024)
    assert direct.call_args.args[1] == limits.cap_kb(2048 * 1024)
    assert not cap.capped
    assert cap.mechanism == "none"


def test_direct_group_is_verified_from_actual_counter_and_limit(tmp_path):
    requested = limits.cap_kb(2048 * 1024)
    group = memory_group(tmp_path / "expman-test", cap=int(requested * 1024))
    with patch.object(limits, "_direct", return_value=group):
        cap = limits.memory_limit("expman-test", 2048 * 1024)
    assert cap.verified and cap.capped
    assert cap.cap_kb == requested


def test_an_attempt_without_its_own_cgroup_reports_process_usage(monkeypatch):
    monkeypatch.setenv("EXPMAN_CGROUP", "")
    snapshot = _worker._resource_snapshot(_worker._resource_cgroup())
    assert snapshot["host_current_kb"] > 0
    assert snapshot["host_peak_kb"] >= snapshot["host_current_kb"]
    assert snapshot["host_usage_source"] == "proc"
    assert snapshot["cgroup"] is None
    assert snapshot["memory_events"] == {}


def test_given_path_must_match_actual_attempt_membership(monkeypatch, tmp_path):
    group = memory_group(tmp_path / "expman-test")
    shared = memory_group(tmp_path / "user.service")
    monkeypatch.setenv("EXPMAN_CGROUP", str(group))
    monkeypatch.setenv("EXPMAN_CGROUP_UNIT", "expman-test")
    with patch.object(_worker, "own_cgroup", return_value=shared):
        assert _worker._resource_cgroup() is None
    with patch.object(_worker, "own_cgroup", return_value=group):
        assert _worker._resource_cgroup() == group


def test_discovery_is_retried_when_scope_becomes_ready(monkeypatch, tmp_path):
    group = memory_group(tmp_path / "expman-test.scope")
    monkeypatch.delenv("EXPMAN_CGROUP", raising=False)
    monkeypatch.setenv("EXPMAN_CGROUP_UNIT", "expman-test")
    with patch.object(
        _worker, "own_cgroup", side_effect=[tmp_path / "user.service", group]
    ):
        assert _worker._resource_cgroup() is None
        assert _worker._resource_cgroup() == group


def test_old_v2_kernel_without_memory_peak_keeps_observed_high_water(tmp_path):
    group = memory_group(tmp_path / "expman-test", peak=None)
    assert limits.usage(group) == (4, 4)
    (group / "memory.current").write_text("2048")
    assert limits.usage(group) == (2, 4)


def test_unreadable_cgroup_falls_back_to_process_with_diagnostics(
    monkeypatch, tmp_path
):
    group = memory_group(tmp_path / "expman-test")
    monkeypatch.setenv("EXPMAN_CGROUP_UNIT", "expman-test")
    monkeypatch.setenv("EXPMAN_HOST_LIMIT_KB", "16")
    with (
        patch.object(_worker, "usage", return_value=None),
        patch.object(_worker, "_own_usage", return_value=(123.0, 456.0)),
    ):
        snapshot = _worker._resource_snapshot(group)
    assert snapshot["host_current_kb"] == 123
    assert snapshot["host_peak_kb"] == 456
    assert snapshot["host_usage_source"] == "proc"
    assert snapshot["host_usage_error"]
    assert not snapshot["host_limit_verified"]


@pytest.mark.parametrize("version,unlimited", [(1, 9223372036854771712), (2, "max")])
def test_unlimited_group_is_explicitly_not_capped(
    monkeypatch, tmp_path, version, unlimited
):
    group = memory_group(tmp_path / "expman-test", version=version, cap=unlimited)
    monkeypatch.setenv("EXPMAN_CGROUP_UNIT", "expman-test")
    monkeypatch.setenv("EXPMAN_HOST_LIMIT_KB", "16")
    snapshot = _worker._resource_snapshot(group)
    assert snapshot["host_limit_kb"] == 0
    assert not snapshot["host_limit_verified"]


def test_larger_than_requested_limit_is_not_claimed_as_enforced(monkeypatch, tmp_path):
    group = memory_group(tmp_path / "expman-test", cap=1024 * 1024)
    monkeypatch.setenv("EXPMAN_CGROUP_UNIT", "expman-test")
    monkeypatch.setenv("EXPMAN_HOST_LIMIT_KB", "16")
    snapshot = _worker._resource_snapshot(group)
    assert snapshot["host_limit_kb"] == 1024
    assert not snapshot["host_limit_verified"]


def test_no_readable_accounting_is_reported_as_unavailable():
    with (
        patch.object(_worker, "_own_usage", return_value=None),
    ):
        snapshot = _worker._resource_snapshot(None)
    assert snapshot["host_current_kb"] is None
    assert snapshot["host_peak_kb"] is None
    assert snapshot["host_usage_source"] == "unavailable"
    assert snapshot["host_usage_error"]


def test_v1_memory_failcnt_is_reported(tmp_path):
    group = memory_group(tmp_path / "expman-test", version=1)
    (group / "memory.failcnt").write_text("2")
    assert limits.refused(limits.events(group))


def test_direct_creation_does_not_remove_preexisting_group(tmp_path):
    parent = memory_group(tmp_path / "session.scope")
    existing = tmp_path / "expman-test"
    existing.mkdir()
    with patch.object(limits, "own_cgroup", return_value=parent):
        assert limits._direct("expman-test", 1024) is None
    assert existing.is_dir()
