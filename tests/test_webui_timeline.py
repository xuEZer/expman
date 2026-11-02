"""Execute the shipped dashboard JavaScript, using the dev Node runtime."""

import json
from pathlib import Path

from nodejs_wheel import node

APP = Path(__file__).parents[1] / "src/expman/webui/static/app.js"
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


def test_serial_attempts_after_one_hundred_seconds_share_one_lane():
    html = render([row("first", 120, 140), row("second", 150, 170)], elapsed=200)

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


def test_cache_reuses_do_not_increase_worker_concurrency():
    html = render(
        [
            row("worker", 0, 200),
            row("reuse1", 120, 120, reused=True),
            row("reuse2", 150, 150, reused=True),
        ]
    )

    assert html.count('class="gantt-lane"') == 1
    assert html.count('class="gantt-bar ') == 1
    assert html.count('class="gantt-reuse"') == 2


def test_reuse_only_history_has_markers_without_concurrent_workers():
    html = render([row("reuse", 120, 120, reused=True)])

    assert 'class="gantt-lane"' not in html
    assert 'class="gantt-bar ' not in html
    assert html.count('class="gantt-reuse"') == 1


def test_unverified_host_limit_has_no_cap_and_reports_usage_fallback():
    html = render_js(
        "workerRow",
        {
            "run": "sample",
            "host": {
                "reserved_kb": 100,
                "resident_kb": None,
                "requested_cap_kb": 50,
                "cap_kb": None,
                "usage_source": "proc",
                "usage_error": "cgroup <unavailable>",
            },
        },
    )

    assert 'class="cap"' not in html
    assert "内存 实际占用 –" in html
    assert "内存采样 进程观测" in html
    assert "内存上限未核验" in html
    assert "cgroup &lt;unavailable&gt;" in html


def test_unknown_host_total_is_not_rendered_as_zero():
    html = render_js(
        "hostRow",
        {
            "total_kb": 64 * 1024 * 1024,
            "expman_actual_kb": None,
            "expman_budget_kb": 8 * 1024 * 1024,
            "other_kb": None,
        },
    )

    assert "实际驻留 观测不可用" in html
    assert "实际驻留 0.0" not in html
    assert "内存观测不可用" in html
