import json

import pytest
from nodejs_wheel import node

from expman.webui.server import STATIC_ROOT


def analyze(data, x, expression):
    program = (
        (STATIC_ROOT / "plot.js").read_text()
        + """
    const [data, x] = JSON.parse(require("node:fs").readFileSync(0, "utf8"));
    const axis = analyzePlotAxis(data, x);
    const metrics = axis.metrics.map(col => col.id);
    const controls = ys => plotControls(axis, ys).map(col => [col.id, col.values.map(plotLabel)]);
    const plot = (ys, choices = []) => buildPlot(axis, ys,
      new Map(choices.map(([id, values]) => [id, new Set(values)]))).series.map(line => line.points);
    """
        + f"process.stdout.write(JSON.stringify({expression}));"
    )
    result = node(
        ["-e", program],
        input=json.dumps([data, x]),
        text=True,
        capture_output=True,
        check=True,
        return_completed_process=True,
    )
    return json.loads(result.stdout)


def columns(*names):
    return [{"id": name, "label": name} for name in names]


def test_each_metric_has_one_filtered_curve_and_only_relevant_configuration_controls():
    metrics = columns("loss", "time", "fixed")
    for col, axes in zip(metrics, [["lr", "seed"], ["lr", "size"], []], strict=True):
        col["depends_on"] = axes
    rows = [
        {
            "lr": lr,
            "seed": seed,
            "size": size,
            "loss": 2 * seed + 1 if lr == 1 else 4,
            "time": lr + size,
            "fixed": 99,
        }
        for lr in (1, 2)
        for seed in (0, 1)
        for size in (10, 20)
    ]
    result = analyze(
        {"variables": columns("lr", "seed", "size"), "metrics": metrics, "rows": rows},
        "lr",
        """({metrics, loss:controls(["loss"]), both:controls(metrics),
        curves:plot(metrics), fixed:plot(metrics, [["seed",[1]], ["size",[20]]]),
        empty:plot(metrics, [["size",[]]])})""",
    )
    assert result["metrics"] == ["loss", "time"]
    assert result["loss"] == [["seed", ["0", "1"]]]
    assert result["both"] == [*result["loss"], ["size", ["10", "20"]]]
    loss, time = result["curves"]
    assert loss == [
        {"x": 1, "y": 2, "deviation": pytest.approx((4 / 3) ** 0.5), "n": 4},
        {"x": 2, "y": 4, "deviation": None, "n": 4},
    ]
    assert [p["y"] for p in time] == pytest.approx([16, 17])
    assert [[p["y"] for p in curve] for curve in result["fixed"]] == [[3, 4], [21, 22]]
    assert all(p["deviation"] is None for curve in result["fixed"] for p in curve)
    assert result["empty"] == []


def test_step_requires_within_run_variation_and_preserves_confounded_controls():
    data = {
        "variables": columns("seed", "model"),
        "metrics": columns("loss", "summary", "flat"),
        "rows": [{"seed": 0, "summary": 1}, {"seed": 1, "model": None, "summary": 2}],
        "steps": [
            {"loss": [[0, 0], [2, 2]], "flat": [[0, 7], [2, 7]]},
            {"loss": [[0, 2], [1, 3]], "flat": [[0, 8], [1, 8]]},
        ],
    }
    result = analyze(
        data,
        "step",
        """({metrics, controls:controls(metrics),
        all:plot(metrics), missing:plot(metrics, [["model",[undefined]]])})""",
    )
    assert result["metrics"] == ["loss"]
    assert result["controls"] == [["seed", ["0", "1"]], ["model", ["–", "null"]]]
    assert result["all"][0] == [
        {"x": 0, "y": pytest.approx(1), "deviation": pytest.approx(2**0.5), "n": 2},
        {"x": 1, "y": 3, "deviation": None, "n": 1},
        {"x": 2, "y": 2, "deviation": None, "n": 1},
    ]
    assert [p["x"] for p in result["missing"][0]] == [0, 2]


def test_metric_axes_align_real_steps_and_preserve_reversals():
    trajectories = {
        "variables": [],
        "metrics": columns("rate", "loss", "unmatched"),
        "rows": [{"rate": 2, "loss": 30, "unmatched": 6}],
        "steps": [
            {
                "rate": [[0, 2], [1, 1], [3, 2]],
                "loss": [[0, 10], [1, 20], [2, 70], [3, 30]],
                "unmatched": [[2, 5], [4, 6]],
            }
        ],
    }
    result = analyze(trajectories, "rate", "({metrics, line:plot(metrics)[0]})")
    assert result["metrics"] == ["loss"]
    assert [(p["x"], p["y"]) for p in result["line"]] == [(2, 10), (1, 20), (2, 30)]
    assert all(p["deviation"] is None for p in result["line"])


def test_cross_run_metric_contrasts_match_settings_and_steps_or_summaries():
    axes = [(a, b) for a in (1, 2) for b in (1, 2)]
    data = {
        "variables": columns("a", "b"),
        "metrics": columns("x", "correlated", "unrelated", "timed", "disjoint"),
        "rows": [
            {
                "a": a,
                "b": b,
                "x": a,
                "correlated": -a,
                "unrelated": b,
                "timed": 3 * a,
                "disjoint": 3 * a,
            }
            for a, b in axes
        ],
        "steps": [
            {"x": [[0, a], [a, a]], "timed": [[0, 3 * a]], "disjoint": [[a, 3 * a]]}
            for a, _ in axes
        ],
    }
    assert analyze(data, "x", "metrics") == ["correlated", "timed"]
