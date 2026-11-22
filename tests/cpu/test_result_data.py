import sqlite3
from contextlib import closing
from datetime import date

import pytest

from expman.observation.results import read_plot, read_results
from expman.storage.base import StorageError, write_record
from expman.storage.batch import BatchStore
from expman.storage.metrics import MetricStore, flatten_metrics
from expman.webui.result_layout import layout_results


def save_configs(directory, configs):
    store = BatchStore(directory)
    entries = []
    for run_id, config in configs:
        entries.append({"run_id": run_id, "attempts": []})
        path = directory / "experiments" / run_id / "config.pkl"
        write_record(path, config, store.serializer)
    store.save_manifest({"experiments": entries, "queue": list(dict(configs))})


def label_rows(columns, rows):
    return [
        {col["label"]: row[col["id"]] for col in columns if col["id"] in row}
        for row in rows
    ]


def test_reads_configs_in_expansion_order_with_the_highest_steps_in_the_snapshot(
    tmp_path,
):
    configs = [
        ("later", {"fixed": 0, "model": {"alpha": 5}, "model.alpha": 2}),
        ("first", {"fixed": 0, "model": {"alpha": 1}, "weight": None}),
        ("waiting", {"fixed": 0, "model": {"alpha": 5}, "model.alpha": 3}),
    ]
    save_configs(tmp_path, configs)
    metrics = MetricStore(tmp_path / "metrics.sqlite3")
    for stage, step, attempt, value in [
        ((0,), 10, 1, 4),
        ((0,), 10, 2, 7),
        ((0,), 9, 2, 99),
        ((0,), None, 2, 100),
        ((1, 0), None, 1, 2),
    ]:
        metrics.log(
            "later", stage, step, attempt, flatten_metrics({"test": {"loss": value}})
        )
    metrics.log("first", (0,), 3, 1, flatten_metrics({"test": {"loss": 5}}))
    metrics.log("first", (0,), 2, 1, flatten_metrics({"test.loss": 0}))
    metrics.log("foreign", (0,), None, 1, flatten_metrics({"other": 8}))

    with closing(sqlite3.connect(metrics.path)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("BEGIN")
        connection.execute("SELECT 1 FROM metrics LIMIT 1").fetchone()
        # A higher step committed after this snapshot belongs to the next read.
        metrics.log("later", (0,), 11, 2, flatten_metrics({"test": {"loss": 12}}))
        variables, outputs, rows = read_results(tmp_path, connection)
    assert label_rows(variables, rows) == [
        {"model.alpha": 5, '["model.alpha"]': 2},
        {"model.alpha": 1, "weight": None},
        {"model.alpha": 5, '["model.alpha"]': 3},
    ]
    assert len(outputs) == 3
    assert label_rows(outputs, rows) == [
        {"S0 · test.loss": 7.0, "S1.0 · test.loss": 2.0},
        {"S0 · test.loss": 5.0, '["test.loss"]': 0.0},
        {},
    ]


def test_preserves_structured_and_typed_yaml_choices(tmp_path):
    save_configs(
        tmp_path,
        [
            (
                "a",
                {
                    "day": date(2026, 1, 1),
                    "features": [1, "two"],
                    "model": {},
                    "id": 2**60,
                },
            ),
            (
                "b",
                {
                    "day": "2026-01-01",
                    "features": ["1", "two"],
                    "model": {"shape": "new"},
                    "id": 2**60 + 1,
                },
            ),
        ],
    )
    variables, outputs, rows = read_results(tmp_path, None)
    assert not outputs and not (tmp_path / "metrics.sqlite3").exists()
    a, b = label_rows(variables, rows)
    assert a["day"] == "2026-01-01" and b["day"] == "'2026-01-01'"
    assert a["features"] == "[1, two]" and b["features"] == "['1', two]"
    assert a["model"] == "{}" and b["model"] == "{shape: new}"
    assert a["id"] == str(2**60) and b["id"] == str(2**60 + 1)


def test_single_config_without_recorded_metrics_still_produces_one_row(tmp_path):
    save_configs(tmp_path, [("one", {"device": [0], "seed": 42})])
    assert layout_results(*read_results(tmp_path, None)) == {
        "columns": [],
        "rows": [{}],
    }


def test_plot_keeps_axis_types_and_gets_only_selected_stages_and_steps(tmp_path):
    configs = [
        ("b", {"rate": 0.25, "typed": 1, "tag": "001", "large": 2**60}),
        ("a", {"rate": 1, "typed": 1.0, "tag": "1", "large": 2**60 + 1}),
        ("waiting", {"rate": 0, "typed": 2, "tag": "2", "large": 2**60 + 2}),
    ]
    save_configs(tmp_path, configs)
    plot = read_plot(tmp_path, None, frozenset())
    assert plot["steps"] == []
    assert label_rows(plot["variables"], plot["rows"]) == [
        {**cfg, "typed": str(cfg["typed"]), "large": str(cfg["large"])}
        for _, cfg in configs
    ]
    metrics = MetricStore(tmp_path / "metrics.sqlite3")
    metrics.log("b", (1, 0), 7, 1, flatten_metrics({"loss": 3}))
    with closing(sqlite3.connect(metrics.path)) as connection:
        (original,) = read_plot(tmp_path, connection, frozenset())["metrics"]
        metrics.log("a", (0,), 3, 1, flatten_metrics({"loss": 100}))
        metrics.log("b", (1, 0), 2, 1, flatten_metrics({"loss": 5}))
        metrics.log("b", (1, 0), 2, 2, flatten_metrics({"loss": 4}))
        metrics.log("b", (1, 0), None, 2, flatten_metrics({"loss": 9}))
        metrics.log("other", (1, 0), 0, 1, flatten_metrics({"loss": 99}))
        plot = read_plot(tmp_path, connection, frozenset([original["id"]]))

    assert {column["label"]: column["id"] for column in plot["metrics"]}[
        "S1.0 · loss"
    ] == original["id"]
    assert label_rows(plot["metrics"], plot["rows"]) == [
        {"S1.0 · loss": 3},
        {"S0 · loss": 100},
        {},
    ]
    assert plot["steps"] == [{original["id"]: [(2, 4), (7, 3)]}, {}, {}]


def test_plot_catalog_identifies_metrics_that_vary_in_controlled_comparisons(tmp_path):
    configs = [
        ("first", {"width": 1, "depth": 1}),
        ("deep", {"width": 1, "depth": 2}),
        ("wide", {"width": 2, "depth": 2}),
    ]
    save_configs(tmp_path, configs)
    store = MetricStore(tmp_path / "metrics.sqlite3")
    for run, config in configs:
        store.log(run, (0,), None, 1, flatten_metrics({"quality": config["depth"]}))
    store.log("deep", (0,), 0, 1, flatten_metrics({"cost": 1}))
    store.log("wide", (0,), 0, 1, flatten_metrics({"cost": 5}))
    with closing(sqlite3.connect(store.path)) as connection:
        plot = read_plot(tmp_path, connection, frozenset())
    axes = {column["id"]: column["label"] for column in plot["variables"]}
    assert {
        metric["label"]: (
            [axes[key] for key in metric["depends_on"]],
            metric["has_steps"],
        )
        for metric in plot["metrics"]
    } == {"quality": (["depth"], False), "cost": (["width"], True)}


def test_unreadable_metrics_are_reported(tmp_path):
    save_configs(tmp_path, [("one", {})])
    path = tmp_path / "metrics.sqlite3"
    path.write_bytes(b"invalid sqlite database")
    with (
        closing(sqlite3.connect(path)) as connection,
        pytest.raises(StorageError, match="cannot read metrics"),
    ):
        read_results(tmp_path, connection)
