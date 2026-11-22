import sqlite3
from contextlib import closing
from datetime import date

import pytest

from expman.observation.results import read_results
from expman.storage.base import StorageError, write_record
from expman.storage.batch import BatchStore
from expman.storage.metrics import MetricStore, flatten_metrics


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


def test_reads_each_config_in_expansion_order_with_its_highest_metric_steps(tmp_path):
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
    assert read_results(tmp_path, None) == ([], [], [{}])


def test_unreadable_metrics_are_reported(tmp_path):
    save_configs(tmp_path, [("one", {})])
    path = tmp_path / "metrics.sqlite3"
    path.write_bytes(b"invalid sqlite database")
    with (
        closing(sqlite3.connect(path)) as connection,
        pytest.raises(StorageError, match="cannot read metrics"),
    ):
        read_results(tmp_path, connection)


def test_results_use_the_callers_metric_snapshot(tmp_path):
    save_configs(tmp_path, [("one", {})])
    metrics = MetricStore(tmp_path / "metrics.sqlite3")
    metrics.log("one", (0,), None, 1, flatten_metrics({"score": 1}))
    with closing(sqlite3.connect(metrics.path)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("BEGIN")
        version = connection.execute("PRAGMA data_version").fetchone()
        metrics.log("one", (0,), None, 1, flatten_metrics({"score": 2}))
        _, columns, rows = read_results(tmp_path, connection)
        assert label_rows(columns, rows) == [{"score": 1.0}]
        connection.commit()
        assert connection.execute("PRAGMA data_version").fetchone() != version
        _, columns, rows = read_results(tmp_path, connection)
        assert label_rows(columns, rows) == [{"score": 2.0}]
