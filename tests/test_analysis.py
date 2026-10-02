import json

import pytest

from expman.analysis import AnalysisError, analysis_payload
from expman.metrics import MetricStore
from expman.storage import PickleSerializer, write_record


@pytest.fixture
def root(tmp_path):
    return tmp_path


def make_batch(root, runs, name="batch-a"):
    """Persist a synthetic batch; each run declares its own rows and metrics.

    ``metric_rows`` items are ``(stage, step, name, value)`` tuples; ``stage``
    is the stage-path tuple and ``name`` the metric-path tuple.
    """
    directory = root / "runs" / name
    experiments = []
    for run in runs:
        exp_dir = directory / "experiments" / run["run_id"]
        exp_dir.mkdir(parents=True, exist_ok=True)
        write_record(exp_dir / "config.pkl", run["config"], PickleSerializer())
        attempts = [
            {
                "attempt": 1,
                "status": run["status"],
                "duration_seconds": run.get("duration"),
                "error_type": None,
                "error_message": None,
            }
        ]
        experiments.append({"run_id": run["run_id"], "attempts": attempts})
    write_record(
        directory / "batch.pkl",
        {"version": 1, "experiments": experiments, "queue": []},
        PickleSerializer(),
    )
    store = MetricStore(directory / "metrics.sqlite3")
    for run in runs:
        for stage, step, metric, value in run.get("metric_rows", []):
            store.log(
                run["run_id"], stage, step, 1, [(json.dumps(list(metric)), value)]
            )
    return directory


def metric(payload, name):
    return next(item for item in payload["metrics"] if item["name"] == name)


def pair(payload, metric_name, field_name):
    return next(
        item
        for item in metric(payload, metric_name)["pairs"]
        if item["field"] == field_name
    )


def test_a_simple_batch_pairs_fields_with_metrics(root):
    directory = make_batch(
        root,
        [
            {
                "run_id": "r1",
                "status": "succeeded",
                "duration": 10.0,
                "config": {"lr": 0.1, "kind": "a", "device": [0]},
                "metric_rows": [((0,), -1, ("score",), 1.0)],
            },
            {
                "run_id": "r2",
                "status": "succeeded",
                "duration": 20.0,
                "config": {"lr": 0.2, "kind": "a", "device": [0]},
                "metric_rows": [((0,), -1, ("score",), 2.0)],
            },
            {
                "run_id": "r3",
                "status": "succeeded",
                "duration": 30.0,
                "config": {"lr": 0.3, "kind": "b", "device": [0]},
                "metric_rows": [((0,), -1, ("score",), 5.0)],
            },
            {
                "run_id": "r4",
                "status": "succeeded",
                "duration": 40.0,
                "config": {"lr": 0.4, "kind": "b", "device": [0]},
                "metric_rows": [((0,), -1, ("score",), 6.0)],
            },
        ],
    )
    payload = analysis_payload(directory, 7)
    assert payload["batch_id"] == 7
    coverage = payload["coverage"]
    assert coverage["runs_total"] == coverage["runs_analyzed"] == 4
    assert coverage["constant_fields"] == ["device"]

    score = metric(payload, "score")
    assert score["n"] == 4 and score["missing"] == 0
    assert score["mean"] == pytest.approx(3.5)
    assert score["min"] == 1.0 and score["max"] == 6.0

    lr = pair(payload, "score", "lr")
    assert lr["kind"] == "numeric" and lr["rho"] == pytest.approx(1.0)
    kind = pair(payload, "score", "kind")
    assert kind["kind"] == "categorical"
    assert kind["groups"][0]["value"] == "b"
    assert kind["groups"][0]["mean"] == pytest.approx(5.5)
    # Exact eta-squared: group means 1.5 and 5.5 around 3.5 -> 16/17.
    assert kind["influence"] == pytest.approx(16 / 17, rel=1e-5)

    duration = metric(payload, "run_duration_seconds")
    assert duration["n"] == 4
    assert pair(payload, "run_duration_seconds", "lr")["rho"] == pytest.approx(1.0)


def test_constant_fields_are_reported_once_and_never_paired(root):
    directory = make_batch(
        root,
        [
            {
                "run_id": f"r{i}",
                "status": "succeeded",
                "config": {"model": {"rank": i, "eps": 0.5}},
                "metric_rows": [((0,), -1, ("score",), float(i))],
            }
            for i in (1, 2, 3)
        ],
    )
    payload = analysis_payload(directory)
    assert payload["coverage"]["constant_fields"] == ["model.eps"]
    assert all(
        item["field"] != "model.eps" for item in metric(payload, "score")["pairs"]
    )
    assert pair(payload, "score", "model.rank")["rho"] == pytest.approx(1.0)


def test_failed_runs_are_excluded_but_counted(root):
    directory = make_batch(
        root,
        [
            {
                "run_id": "ok1",
                "status": "succeeded",
                "config": {"lr": 1},
                "metric_rows": [((0,), -1, ("score",), 1.0)],
            },
            {
                "run_id": "ok2",
                "status": "succeeded",
                "config": {"lr": 2},
                "metric_rows": [((0,), -1, ("score",), 2.0)],
            },
            {
                "run_id": "ok3",
                "status": "succeeded",
                "config": {"lr": 3},
                "metric_rows": [((0,), -1, ("score",), 3.0)],
            },
            {
                "run_id": "bad",
                "status": "failed",
                "config": {"lr": 9},
                "metric_rows": [((0,), -1, ("score",), 99.0)],
            },
        ],
    )
    payload = analysis_payload(directory)
    assert payload["coverage"]["runs_analyzed"] == 3
    assert payload["coverage"]["excluded"] == {"failed": 1}
    score = metric(payload, "score")
    assert score["n"] == 3 and score["max"] == 3.0
    assert pair(payload, "score", "lr")["n"] == 3


def test_missing_metrics_are_counted_per_metric(root):
    directory = make_batch(
        root,
        [
            {
                "run_id": f"r{i}",
                "status": "succeeded",
                "duration": 10.0 * i,
                "config": {"lr": i},
                "metric_rows": [((0,), -1, ("score",), float(i))],
            }
            for i in (1, 2, 3)
        ]
        + [
            {
                "run_id": "r4",
                "status": "succeeded",
                "config": {"lr": 4},
                "metric_rows": [],  # this run never logged anything
            }
        ],
    )
    payload = analysis_payload(directory)
    score = metric(payload, "score")
    assert score["n"] == 3 and score["missing"] == 1
    # The duration pseudo-metric covers only the runs that report one.
    duration = metric(payload, "run_duration_seconds")
    assert duration["n"] == 3 and duration["missing"] == 1


def test_an_unreadable_config_is_skipped_with_its_reason(root):
    directory = make_batch(
        root,
        [
            {
                "run_id": f"r{i}",
                "status": "succeeded",
                "config": {"lr": i},
                "metric_rows": [((0,), -1, ("score",), float(i))],
            }
            for i in (1, 2, 3, 4)
        ],
    )
    (directory / "experiments" / "r4" / "config.pkl").unlink()
    payload = analysis_payload(directory)
    assert [item["run_id"] for item in payload["coverage"]["skipped"]] == ["r4"]
    assert "r4" in payload["coverage"]["skipped"][0]["reason"]
    assert payload["coverage"]["runs_total"] == 4
    assert payload["coverage"]["runs_analyzed"] == 3


def test_a_summary_value_beats_intermediate_steps(root):
    directory = make_batch(
        root,
        [
            {
                "run_id": "r1",
                "status": "succeeded",
                "config": {"lr": 1},
                "metric_rows": [
                    ((0,), 0, ("score",), 1.0),
                    ((0,), 5, ("score",), 2.0),
                    ((0,), -1, ("score",), 0.5),
                ],
            },
            {
                "run_id": "r2",
                "status": "succeeded",
                "config": {"lr": 2},
                "metric_rows": [
                    ((0,), 0, ("score",), 1.0),
                    ((0,), 7, ("score",), 3.0),
                ],
            },
            {
                "run_id": "r3",
                "status": "succeeded",
                "config": {"lr": 3},
                "metric_rows": [
                    ((0,), 0, ("score",), 1.0),
                    ((0,), 9, ("score",), 6.0),
                ],
            },
        ],
    )
    payload = analysis_payload(directory)
    # r1 keeps its summary 0.5; the others keep their last step.
    assert pair(payload, "score", "lr")["rho"] == pytest.approx(1.0)
    score = metric(payload, "score")
    assert score["min"] == 0.5 and score["max"] == 6.0


def test_the_same_metric_name_in_two_stages_is_disambiguated(root):
    directory = make_batch(
        root,
        [
            {
                "run_id": "r1",
                "status": "succeeded",
                "config": {"lr": 1},
                "metric_rows": [
                    ((0,), -1, ("score",), 1.0),
                    ((1,), -1, ("score",), 100.0),
                ],
            },
            {
                "run_id": "r2",
                "status": "succeeded",
                "config": {"lr": 2},
                "metric_rows": [
                    ((0,), -1, ("score",), 2.0),
                    ((1,), -1, ("score",), 200.0),
                ],
            },
            {
                "run_id": "r3",
                "status": "succeeded",
                "config": {"lr": 3},
                "metric_rows": [
                    ((0,), -1, ("score",), 3.0),
                    ((1,), -1, ("score",), 300.0),
                ],
            },
        ],
    )
    payload = analysis_payload(directory)
    names = {item["name"] for item in payload["metrics"]}
    assert names == {"S0.score", "S1.score"}


def test_a_batch_without_metrics_still_reports_the_table(root):
    directory = make_batch(
        root,
        [
            {"run_id": "r1", "status": "succeeded", "config": {"lr": 1}},
            {"run_id": "r2", "status": "failed", "config": {"lr": 2}},
        ],
    )
    payload = analysis_payload(directory)
    assert payload["coverage"]["runs_total"] == 2
    assert payload["coverage"]["runs_analyzed"] == 1
    assert payload["coverage"]["excluded"] == {"failed": 1}
    assert payload["metrics"] == []
    assert payload["coverage"]["constant_fields"] == ["lr"]


def test_pairs_with_too_few_points_are_counted_not_shown(root):
    directory = make_batch(
        root,
        [
            {
                "run_id": "r1",
                "status": "succeeded",
                "config": {"lr": 1, "kind": "a"},
                "metric_rows": [((0,), -1, ("score",), 1.0)],
            },
            {
                "run_id": "r2",
                "status": "succeeded",
                "config": {"lr": 2, "kind": "b"},
                "metric_rows": [((0,), -1, ("score",), 2.0)],
            },
        ],
    )
    payload = analysis_payload(directory)
    score = metric(payload, "score")
    assert score["pairs"] == []
    assert score["skipped_fields"] == 2
    assert score["influence_max"] is None


def test_a_flat_metric_and_a_lonely_group_are_annotated(root):
    directory = make_batch(
        root,
        [
            {
                "run_id": f"r{i}",
                "status": "succeeded",
                "config": {"lr": i, "kind": chr(96 + i)},
                "metric_rows": [
                    ((0,), -1, ("flat",), 2.0),
                    ((0,), -1, ("solo",), float(i)),
                ],
            }
            for i in (1, 2, 3)
        ],
    )
    payload = analysis_payload(directory)
    flat_lr = pair(payload, "flat", "lr")
    assert flat_lr["kind"] == "numeric"
    assert flat_lr["rho"] is None and flat_lr["influence"] is None
    assert "无变异" in flat_lr["note"]

    solo_kind = pair(payload, "solo", "kind")
    assert solo_kind["kind"] == "categorical"
    assert solo_kind["influence"] is None
    assert "一个样本" in solo_kind["note"]
    assert solo_kind["best"] == "c"
    # The numeric field still gets a real correlation on the varying metric.
    assert pair(payload, "solo", "lr")["rho"] == pytest.approx(1.0)


def test_a_reversed_trend_gets_a_negative_rank_correlation(root):
    directory = make_batch(
        root,
        [
            {
                "run_id": f"r{i}",
                "status": "succeeded",
                "config": {"lr": i},
                "metric_rows": [((0,), -1, ("score",), 4.0 - i)],
            }
            for i in (1, 2, 3, 4)
        ],
    )
    assert pair(analysis_payload(directory), "score", "lr")["rho"] == pytest.approx(
        -1.0
    )


def test_a_directory_without_a_manifest_is_rejected(root):
    with pytest.raises(AnalysisError):
        analysis_payload(root / "runs" / "nothing")


def test_two_stage_names_and_lists_are_categorical_leaves(root):
    directory = make_batch(
        root,
        [
            {
                "run_id": "r1",
                "status": "succeeded",
                "config": {"devices": [0], "tag": "x"},
                "metric_rows": [((0,), -1, ("score",), 1.0)],
            },
            {
                "run_id": "r2",
                "status": "succeeded",
                "config": {"devices": [0], "tag": "x"},
                "metric_rows": [((0,), -1, ("score",), 1.0)],
            },
            {
                "run_id": "r3",
                "status": "succeeded",
                "config": {"devices": [1], "tag": "x"},
                "metric_rows": [((0,), -1, ("score",), 5.0)],
            },
        ],
    )
    payload = analysis_payload(directory)
    devices = pair(payload, "score", "devices")
    assert devices["kind"] == "categorical"
    assert devices["groups"][0]["value"] == [1]
    assert payload["coverage"]["constant_fields"] == ["tag"]
