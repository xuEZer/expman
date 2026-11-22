"""Join a batch's expanded configurations to its durable scalar metrics."""

import json
import math
import sqlite3
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from expman.storage.base import StorageError
from expman.storage.batch import BatchStore

type _Path = tuple[str, ...]
type _Cell = str | int | float | bool | None
type _Row = dict[str, _Cell]
type _Column = dict[str, str]


def _label(path: _Path) -> str:
    """Use dotted paths, quoting literal keys that could look like paths."""
    parts = []
    for key in path:
        if key.isidentifier():
            parts.append(("." if parts else "") + key)
        else:
            parts.append(f"[{json.dumps(key, ensure_ascii=False)}]")
    return "".join(parts)


def _fields(
    mappings: dict[int, dict], parent: _Path = ()
) -> Iterator[tuple[_Path, dict[int, Any]]]:
    children: dict[str, dict[int, Any]] = {}
    for index, mapping in mappings.items():
        for key, value in mapping.items():
            children.setdefault(key, {})[index] = value
    for key, cells in children.items():
        path = (*parent, key)
        if all(isinstance(value, dict) and value for value in cells.values()):
            yield from _fields(cells, path)
        else:
            # A list, an empty mapping or a mix of mappings and scalars is
            # one field. Flattening it could erase the choice of an empty value.
            yield path, cells


class _ValueDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        # YAML aliases have already been resolved in these acyclic config values.
        return True


def _cells(values: dict[int, Any]) -> dict[int, _Cell]:
    """Send ordinary scalars directly and preserve richer YAML types as text."""
    non_null = [value for value in values.values() if value is not None]
    types = {type(value) for value in non_null}
    if len(types) <= 1 and (
        types <= {str, bool}
        or (types == {int} and all(abs(value) < 2**53 for value in non_null))
        or (types == {float} and all(math.isfinite(value) for value in non_null))
    ):
        return values
    # Encode the entire column together so a timestamp and its spelling as a
    # quoted string, for example, stay distinct all the way to the browser.
    return {
        index: None
        if value is None
        else yaml.dump(
            value, Dumper=_ValueDumper, allow_unicode=True, default_flow_style=True
        )
        .removesuffix("\n...\n")
        .rstrip("\n")
        for index, value in values.items()
    }


def _variables(configs: dict[int, dict], rows: list[_Row]) -> list[_Column]:
    columns: list[_Column] = []
    for path, values in _fields(configs):
        cells = _cells(values)
        if len(cells) == len(rows) and len(set(cells.values())) == 1:
            continue
        column_id = f"x{len(columns)}"
        columns.append({"id": column_id, "label": _label(path)})
        for index, value in cells.items():
            rows[index][column_id] = value
    return columns


def _metrics(
    connection: sqlite3.Connection | None, rows: dict[str, _Row]
) -> list[_Column]:
    if connection is None:
        return []
    keys: dict[tuple[tuple[int, ...], _Path], str] = {}
    stages: dict[_Path, set[tuple[int, ...]]] = defaultdict(set)
    try:
        if not connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='metrics'"
        ).fetchone():
            return []
        # Summary metrics use step -1, below every explicit step.
        for run_id, stage_json, metric_json, step, value in connection.execute(
            """WITH latest AS (
                   SELECT run_id, stage_path, metric_path, MAX(step) AS step
                   FROM metrics
                   GROUP BY run_id, stage_path, metric_path
               )
               SELECT run_id, stage_path, metric_path, step, value
               FROM latest JOIN metrics USING (run_id, stage_path, metric_path, step)
               ORDER BY run_id, stage_path, metric_path"""
        ):
            if run_id not in rows:
                continue
            stage, metric = (
                tuple(json.loads(stage_json)),
                tuple(json.loads(metric_json)),
            )
            if (
                not stage
                or not all(type(i) is int and i >= 0 for i in stage)
                or not metric
                or not all(isinstance(k, str) for k in metric)
                or type(step) is not int
                or step < -1
                or not math.isfinite(value)
            ):
                raise ValueError("invalid metric record")
            key = (stage, metric)
            if key not in keys:
                keys[key] = f"y{len(keys)}"
                stages[metric].add(stage)
            rows[run_id][keys[key]] = value
    except (sqlite3.Error, ValueError, TypeError) as error:
        raise StorageError(f"cannot read metrics: {error}") from error

    columns = []
    for stage, metric in sorted(keys):
        label = _label(metric)
        if len(stages[metric]) > 1:
            label = f"S{'.'.join(map(str, stage))} · {label}"
        columns.append({"id": keys[(stage, metric)], "label": label})
    return columns


def read_results(
    directory: Path, connection: sqlite3.Connection | None
) -> tuple[list[_Column], list[_Column], list[_Row]]:
    """Read one sparse row per manifest run, in YAML expansion order.

    Missing cells have no key. Each stage and metric path supplies one column,
    using the highest recorded step for each run in the caller's database
    snapshot. Pass None until the first metric database exists.
    """
    store = BatchStore(directory)
    entries = store.read_manifest()["experiments"]
    configs: dict[int, dict] = {}
    by_run: dict[str, _Row] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise StorageError("invalid experiment entry in batch manifest")
        run_id = entry.get("run_id")
        if not isinstance(run_id, str) or run_id in by_run:
            raise StorageError("invalid or duplicate experiment ID in batch manifest")
        configs[index] = store.load_config(run_id)
        by_run[run_id] = {}
    rows = list(by_run.values())
    variables = _variables(configs, rows)
    metrics = _metrics(connection, by_run)
    return variables, metrics, rows
