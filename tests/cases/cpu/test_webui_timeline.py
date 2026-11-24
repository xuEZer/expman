"""Execute the shipped dashboard JavaScript, using the dev Node runtime."""

import json

from nodejs_wheel import node

from expman.webui.server import STATIC_ROOT

APP = STATIC_ROOT / "app.js"
HARNESS = r"""
const fs = require("node:fs");
const vm = require("node:vm");
const context = vm.createContext({
  document: {body: {dataset: {page: "batch"}}, getElementById: () => ({})},
  location: {pathname: "/b/0"},
  fetch: () => new Promise(() => {}),
  EventSource: function () {},
});
vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), context);
const payload = JSON.parse(fs.readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify(context[process.argv[2]](payload)));
"""


def render_js(function, payload):
    result = node(
        ["-e", HARNESS, str(APP), function],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
        return_completed_process=True,
    )
    return json.loads(result.stdout)


def render(rows, *, elapsed=300.0, marks=()):
    return render_js(
        "timeline",
        {"gantt": rows, "batch": {"elapsed_seconds": elapsed, "session_marks": marks}},
    )


def row(run, start, end, *, reused=False, status="succeeded", attempt=1):
    return {
        "run": run,
        "stage_index": 0,
        "stage": "Work",
        "attempt": attempt,
        "start_seconds": start,
        "end_seconds": end,
        "status": status,
        "reused": reused,
    }


def test_serial_attempts_share_one_lane_and_cache_hits_do_not_extend_the_axis():
    html = render(
        [
            row("first", 120, 140),
            row("second", 150, 170),
            row("reuse", 350, 350, reused=True),
        ],
        elapsed=200,
    )

    assert html.count('class="gantt-lane"') == 1
    assert html.count('class="gantt-bar ') == 2
    assert "left:60%;width:10%" in html
    assert "left:75%;width:10%" in html


def test_overlaps_and_a_running_retry_use_their_real_lanes():
    html = render(
        [
            row("retry", 120, 180, status="cancelled"),
            row("other", 150, 200),
            row("retry", 210, 300, status="running", attempt=2),
        ],
        marks=[205],
    )

    assert html.count('class="gantt-lane"') == 2
    assert html.count('class="gantt-bar ') == 3
    assert 'class="gantt-bar warn"' in html
    assert "retry · Work a2" in html
    assert "left:70%;width:30%" in html
    assert 'class="gantt-fold"' in html


def test_reuse_only_history_shows_the_empty_timeline():
    assert "暂无带起止时刻的尝试记录" in render([row("reuse", 120, 120, reused=True)])


def test_worker_bars_scale_usage_to_the_assigned_limits():
    gib = 1024**2
    html = render_js(
        "workers",
        {
            "workers": [
                {
                    "host": {
                        "resident_kb": 4 * gib,
                        "reserved_kb": 2 * gib,
                        "requested_cap_kb": 8 * gib,
                        "cap_kb": 6 * gib,
                    },
                    "gpu": {
                        "used_kb": 5 * gib,
                        "budget_kb": 3 * gib,
                        "requested_cap_kb": 4 * gib,
                        "cap_kb": None,
                    },
                }
            ],
        },
    )

    assert "<span>内存</span><span>显存</span>" in html
    assert html.count('class="worker-meter"') == 2
    assert html.count("width:100%;background:var(--other)") == 2
    assert "width:50%;background:var(--actual)" in html
    assert "width:100%;background:var(--actual)" in html
    assert html.count('class="memory-mark worker-allocation" style="--at:100%') == 2
    assert html.count("4.0 GiB") == 2
    assert "5.0 GiB" in html and "8.0 GiB" in html
    assert all(label not in html for label in ("2.0 GiB", "3.0 GiB", "6.0 GiB"))


def test_missing_worker_values_display_only_the_known_quantity():
    gib = 1024**2
    html = render_js(
        "workerRow",
        {
            "host": {"resident_kb": gib, "reserved_kb": 3 * gib, "requested_cap_kb": 0},
            "gpu": {"used_kb": None, "requested_cap_kb": 2 * gib},
        },
    )

    assert html.count('class="track ') == 1
    assert "background:var(--actual)" not in html
    assert html.count("1.0 GiB") == html.count("2.0 GiB") == 1
    assert "0.0 GiB" not in html and "3.0 GiB" not in html


def test_unknown_memory_and_utilization_are_not_rendered_as_zero():
    html = render_js(
        "resources",
        {"resources": {"host": {"total_kb": 64 * 1024**2, "other_kb": None}}},
    )

    assert " GiB" not in html
    assert 'class="resource-utilization mono">–</span>' in html


def test_resource_bars_sum_allocations_for_the_host_and_each_gpu():
    gib = 1024**2
    card = {
        "index": 0,
        "total_kb": 100 * gib,
        "other_kb": 20 * gib,
        "expman_actual_kb": 10 * gib,
        "expman_budget_kb": 15 * gib,
        "utilization_percent": 43,
    }
    html = render_js(
        "resources",
        {
            "workers": [
                {
                    "device": device,
                    "host": {"requested_cap_kb": host * gib},
                    "gpu": {"requested_cap_kb": gpu * gib},
                }
                for device, host, gpu in [(0, 8, 17), (0, 14, 23), (1, 6, 5)]
            ],
            "resources": {
                "devices": [card, card | {"index": 1}],
                "host": card | {"utilization_percent": 0},
            },
        },
    )
    rows = html.split('class="row resource-row"')[1:]
    assert len(rows) == 3
    for row_html, amount in zip(rows, (40, 5, 28), strict=True):
        assert f'class="env" style="left:20%;width:{amount}%' in row_html
        assert (
            f'class="memory-mark resource-allocation" style="--at:{20 + amount}%'
            in row_html
        )
        assert f">{amount}.0 GiB</span>" in row_html
    assert "left:20%;width:10%" in html
    assert 'class="memory-mark resource-other" style="--at:20%' in html
    assert 'class="memory-mark resource-used" style="--at:30%' in html
    assert 'class="resource-utilization mono">43%</span>' in html
    assert 'class="resource-utilization mono">0%</span>' in html


def test_resource_envelope_handles_overcommit_and_incomplete_allocations():
    gib = 1024**2
    reading = {"total_kb": 100 * gib, "other_kb": 80 * gib, "expman_actual_kb": gib}
    html = render_js(
        "resources",
        {
            "workers": [
                {"device": 0, "gpu": {"requested_cap_kb": 30 * gib}},
                {"device": 1, "host": {"requested_cap_kb": gib}},
                {"device": 1, "gpu": {"requested_cap_kb": gib}},
            ],
            "resources": {
                "devices": [reading | {"index": n} for n in (0, 1)],
                "host": reading,
            },
        },
    )
    assert html.count('class="env"') == 1
    assert 'class="env" style="left:80%;width:20%' in html
    assert html.count('class="memory-mark resource-allocation"') == 1
    assert 'class="memory-mark resource-allocation" style="--at:100%' in html
    assert ">30.0 GiB</span>" in html


def test_stale_snapshot_retains_live_group_counts_and_process_state():
    snapshot = {
        "running": True,
        "source": "last",
        "age_seconds": 61,
        "stages": [
            {
                "name": "fit",
                "known": True,
                "approximate": False,
                "groups_total": 2,
                "groups_done": 1,
                "groups_running": 1,
                "groups_pending": 0,
            },
        ],
    }
    assert "批次进程仍在运行" in render_js("sourceNote", snapshot)
    assert render_js("sourceNote", snapshot | {"running": False}) == (
        '<div class="notice warn">批次已停止。</div>'
    )
    assert ">1/1/0<" in render_js("stages", snapshot)
    assert ">1/0/0<" in render_js("stages", snapshot | {"running": False})


def test_dashboard_eta_uses_the_same_outward_rounding_as_the_cli():
    from expman import TimeEstimate

    eta = TimeEstimate(61, 119, 0.8, 1, 1)
    html = render_js(
        "tiles", {"batch": {"remaining": {"lower_seconds": 61, "upper_seconds": 119}}}
    )
    assert str(eta) in html


def test_event_details_keep_attempts_outcomes_and_reasons():
    attempt = {"run": "abc123", "stage_index": 0, "attempt": 2, "device": 0}
    events = [
        {**attempt, "kind": "launch", "detail": "host 512M gpu 256M MPS 请求上限 282M"},
        {**attempt, "kind": "finish", "status": "failed", "detail": "2.7s 峰值 43M"},
        {"kind": "reuse", "detail": "等价配置命中前缀缓存"},
        {"kind": "shed", "detail": "设备 0 显存不足以兑现已分配的配额，撤销最新的尝试"},
    ]
    html = render_js("events", {"events": events})

    assert html.count("abc123 S0 a2 GPU0") == 2
    assert "failed" in html and "2.7s" in html
    assert events[2]["detail"] in html and events[3]["detail"] in html
    assert all(field not in html for field in ("host 512M", "gpu 256M", "MPS", "峰值"))
