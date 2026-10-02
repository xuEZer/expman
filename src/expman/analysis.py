"""Offline analysis of a batch's results: simple independent-to-dependent reads.

The module reads only what a batch leaves on disk: the manifest for the run
list, attempt statuses and durations, each run's ``config.pkl`` for the
independent variables, and ``metrics.sqlite3`` for the dependent ones.  It
never imports user code, never executes a pipeline and never writes.

A batch does not have to be complete.  Every metric is summarised over the
runs where it exists, every configuration leaf is paired with every metric
over the runs where both exist, and anything that cannot be read is reported
as a skipped entry with its reason instead of failing the whole analysis.

The methods are deliberately simple.  A numeric field with three or more
distinct values gets a Spearman rank correlation; anything else is grouped by
value, and the share of the metric's total variation that lies between groups
measures the field's influence.  Constant fields are reported once and then
left out of every table.  Nothing here asserts causality: a large influence
only describes the observed data.
"""

import json
import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any

from .storage import PickleSerializer, StorageError, read_record

ANALYSIS_SCHEMA = 1
SUMMARY_STEP = -1  # MetricStore's sentinel for run-level summary values
MIN_NUMERIC_POINTS = 3  # a rank correlation below this says nothing
MIN_GROUPS = 2
DURATION_METRIC = "run_duration_seconds"


class AnalysisError(RuntimeError):
    """The batch directory does not hold readable analysis inputs."""


@dataclass(frozen=True)
class RunRow:
    """One run's independent variables, its metric values and its cost."""

    run_id: str
    status: str
    duration_seconds: float | None
    config: dict
    metrics: dict[str, float]


@dataclass(frozen=True)
class Collected:
    """The joined table plus everything that could not be joined."""

    runs: list[RunRow]
    skipped: list[dict[str, str]]


def fingerprint(directory: Path) -> tuple:
    """A cheap change marker for the inputs one analysis depends on."""
    parts: list[Any] = []
    for name in ("batch.pkl", "metrics.sqlite3"):
        path = directory / name
        try:
            parts.append(path.stat().st_mtime_ns)
        except OSError:
            parts.append(None)
    experiments = directory / "experiments"
    entries: list[tuple[str, int]] = []
    try:
        for child in experiments.iterdir():
            config = child / "config.pkl"
            try:
                entries.append((child.name, config.stat().st_mtime_ns))
            except OSError:
                entries.append((child.name, -1))
    except OSError:
        pass
    parts.append(tuple(sorted(entries)))
    return tuple(parts)


def _latest_status(entry: dict) -> str:
    attempts = entry.get("attempts")
    if isinstance(attempts, list) and attempts:
        last = attempts[-1]
        if isinstance(last, dict) and isinstance(last.get("status"), str):
            return last["status"]
    return "pending"


def _total_seconds(entry: dict) -> float | None:
    attempts = entry.get("attempts")
    if not isinstance(attempts, list):
        return None
    total = 0.0
    seen = False
    for attempt in attempts:
        value = attempt.get("duration_seconds") if isinstance(attempt, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total += float(value)
            seen = True
    return total if seen else None


def _read_config(directory: Path, run_id: str) -> dict:
    path = directory / "experiments" / run_id / "config.pkl"
    try:
        config = read_record(path, PickleSerializer())
    except (StorageError, OSError) as error:
        raise AnalysisError(f"cannot read {path}: {error}") from error
    if not isinstance(config, dict):
        raise AnalysisError(f"{path} does not hold a configuration mapping")
    return config


def _metric_name(stage: tuple, metric: tuple, shared: set[tuple]) -> str:
    if metric in shared:
        return "S" + ".".join(str(part) for part in stage) + "." + ".".join(metric)
    return ".".join(metric)


def _read_metrics(directory: Path, run_ids: set[str]) -> dict[str, dict[str, float]]:
    """Per-run ``{metric name: value}`` maps; summaries beat intermediate steps.

    A metric logged once under the same name in several stages is kept under a
    stage-prefixed name so the two never overwrite each other.
    """
    path = directory / "metrics.sqlite3"
    if not path.exists():
        return {}
    try:
        with closing(sqlite3.connect(path, timeout=30)) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if "metrics" not in tables:
                return {}
            rows = connection.execute(
                "SELECT run_id, stage_path, metric_path, step, value FROM metrics"
            ).fetchall()
    except sqlite3.Error as error:
        raise AnalysisError(f"cannot read {path}: {error}") from error

    parsed: list[tuple[str, tuple, tuple, int, float]] = []
    stages_by_metric: dict[tuple, set[tuple]] = {}
    for run_id, stage_path, metric_path, step, value in rows:
        if run_id not in run_ids or isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        try:
            stage = tuple(json.loads(stage_path))
            metric = tuple(json.loads(metric_path))
        except (TypeError, ValueError):
            continue
        if not metric or not all(isinstance(part, str) for part in metric):
            continue
        parsed.append((run_id, stage, metric, int(step), float(value)))
        stages_by_metric.setdefault(metric, set()).add(stage)

    shared = {metric for metric, stages in stages_by_metric.items() if len(stages) > 1}
    chosen: dict[tuple[str, tuple, tuple], tuple[int, float]] = {}
    for run_id, stage, metric, step, value in parsed:
        key = (run_id, stage, metric)
        current = chosen.get(key)
        # A summary row wins outright; otherwise the largest step is the last.
        if (
            current is None
            or step == SUMMARY_STEP
            or (current[0] != SUMMARY_STEP and step > current[0])
        ):
            chosen[key] = (step, value)

    per_run: dict[str, dict[str, float]] = {run_id: {} for run_id in run_ids}
    for (run_id, stage, metric), (_, value) in chosen.items():
        per_run[run_id][_metric_name(stage, metric, shared)] = value
    return per_run


def collect_runs(directory: Path) -> Collected:
    """Join the manifest, each run's config and the metric store into rows."""
    try:
        manifest = read_record(directory / "batch.pkl", PickleSerializer())
    except (StorageError, OSError) as error:
        raise AnalysisError(
            f"cannot read {directory / 'batch.pkl'}: {error}"
        ) from error
    entries = manifest.get("experiments") if isinstance(manifest, dict) else None
    if not isinstance(entries, list):
        raise AnalysisError(f"{directory} does not hold a Batch manifest")

    run_ids: set[str] = set()
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("run_id"), str):
            run_ids.add(entry["run_id"])
    metrics = _read_metrics(directory, run_ids)

    runs: list[RunRow] = []
    skipped: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("run_id"), str):
            continue
        run_id = entry["run_id"]
        status = _latest_status(entry)
        try:
            config = _read_config(directory, run_id)
        except AnalysisError as error:
            skipped.append({"run_id": run_id, "reason": str(error)})
            continue
        runs.append(
            RunRow(
                run_id=run_id,
                status=status,
                duration_seconds=_total_seconds(entry),
                config=config,
                metrics=metrics.get(run_id, {}),
            )
        )
    return Collected(runs=runs, skipped=skipped)


def _flatten(
    value: Any, path: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], Any]]:
    """The leaves of a configuration tree; lists count as one leaf value."""
    if isinstance(value, dict) and value:
        rows: list[tuple[tuple[str, ...], Any]] = []
        for key, child in value.items():
            rows.extend(_flatten(child, (*path, str(key))))
        return rows
    return [(path, value)]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _group_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _ranks(values: list[float]) -> list[float]:
    """Average ranks, so tied observations share their places."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        average = (start + end) / 2 + 1
        for position in range(start, end + 1):
            ranks[order[position]] = average
        start = end + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    count = len(xs)
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    variance_y = sum((y - mean_y) ** 2 for y in ys)
    if variance_x <= 0 or variance_y <= 0:
        return None
    coefficient = covariance / math.sqrt(variance_x * variance_y)
    return max(-1.0, min(1.0, coefficient))


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    return _pearson(_ranks(xs), _ranks(ys))


def _sample_sd(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _eta_squared(groups: dict[str, list[float]]) -> float | None:
    """The share of the metric's total variation that lies between groups."""
    everything = [value for values in groups.values() for value in values]
    total_mean = sum(everything) / len(everything)
    ss_total = sum((value - total_mean) ** 2 for value in everything)
    if ss_total <= 0:
        return None
    ss_between = 0.0
    for values in groups.values():
        mean = sum(values) / len(values)
        ss_between += len(values) * (mean - total_mean) ** 2
    return ss_between / ss_total


def _numeric_pair(field_name: str, xs: list[float], ys: list[float]) -> dict[str, Any]:
    rho = _spearman(xs, ys)
    note = None
    influence = None if rho is None else abs(rho)
    if rho is None:
        note = "自变量或指标在有效样本内无变异，无法计算秩相关"
    return {
        "field": field_name,
        "kind": "numeric",
        "n": len(xs),
        "rho": rho,
        "influence": influence,
        "note": note,
    }


def _categorical_pair(
    field_name: str, groups: dict[str, list[float]], raw_values: dict[str, Any]
) -> dict[str, Any]:
    summary = []
    for key, values in groups.items():
        summary.append(
            {
                "value": raw_values[key],
                "n": len(values),
                "mean": sum(values) / len(values),
            }
        )
    summary.sort(key=lambda item: (-item["mean"], str(item["value"])))
    lonely = all(len(values) == 1 for values in groups.values())
    effect = None if lonely else _eta_squared(groups)
    if lonely:
        note = "每个取值只有一个样本，只列各组均值，不计算影响力度量"
    else:
        note = None if _eta_squared(groups) is not None else "指标在有效样本内无变异"
    return {
        "field": field_name,
        "kind": "categorical",
        "n": sum(len(values) for values in groups.values()),
        "influence": effect,
        "groups": summary,
        "best": summary[0]["value"] if summary else None,
        "note": note,
    }


def _pair_for(
    field_name: str,
    field_values: list[Any],
    metric_values: list[float],
) -> dict[str, Any] | None:
    """One (field, metric) pair, or None when there is nothing to compare."""
    distinct = {_group_key(value) for value in field_values}
    if len(distinct) <= 1:
        return None
    if all(_is_number(value) for value in field_values) and len(distinct) >= 3:
        return _numeric_pair(
            field_name, [float(value) for value in field_values], metric_values
        )
    groups: dict[str, list[float]] = {}
    raw: dict[str, Any] = {}
    for value, y in zip(field_values, metric_values, strict=True):
        key = _group_key(value)
        groups.setdefault(key, []).append(y)
        raw[key] = value
    if len(groups) < MIN_GROUPS:
        return None
    return _categorical_pair(field_name, groups, raw)


def _metric_stats(values: list[float], analyzed: int) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": sum(values) / len(values) if values else None,
        "sd": _sample_sd(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "missing": analyzed - len(values),
    }


def analysis_payload(directory: Path, batch_id: int | None = None) -> dict:
    """The full analysis of one batch directory, ready for JSON encoding."""
    collected = collect_runs(directory)
    succeeded = [run for run in collected.runs if run.status == "succeeded"]
    statuses: dict[str, int] = {}
    for run in collected.runs:
        statuses[run.status] = statuses.get(run.status, 0) + 1

    # One flattened view per analysed run; every pair reuses it.
    flat: dict[str, dict[tuple[str, ...], Any]] = {
        run.run_id: dict(_flatten(run.config)) for run in succeeded
    }
    field_paths: set[tuple[str, ...]] = set()
    for leaves in flat.values():
        field_paths.update(leaves)

    constant_fields: list[str] = []
    varying: list[tuple[str, ...]] = []
    for path in sorted(field_paths):
        seen = {_group_key(leaves[path]) for leaves in flat.values() if path in leaves}
        if len(seen) <= 1:
            constant_fields.append(".".join(path) if path else "(root)")
        else:
            varying.append(path)

    durations = {
        run.run_id: run.duration_seconds
        for run in succeeded
        if run.duration_seconds is not None
    }
    metric_names: set[str] = {DURATION_METRIC} if durations else set()
    for run in succeeded:
        metric_names.update(run.metrics)

    metric_reports = []
    for name in sorted(metric_names):
        samples: dict[str, float] = {}
        for run in succeeded:
            value = (
                durations.get(run.run_id)
                if name == DURATION_METRIC
                else run.metrics.get(name)
            )
            if value is not None:
                samples[run.run_id] = value
        stats = _metric_stats(list(samples.values()), len(succeeded))

        pairs: list[dict[str, Any]] = []
        insufficient = 0
        for path in varying:
            field_name = ".".join(path)
            picked = [
                (leaves[path], samples[run_id])
                for run_id, leaves in flat.items()
                if run_id in samples and path in leaves
            ]
            if len(picked) < MIN_NUMERIC_POINTS:
                insufficient += 1
                continue
            pair = _pair_for(
                field_name,
                [x for x, _ in picked],
                [y for _, y in picked],
            )
            if pair is None:
                insufficient += 1
            else:
                pairs.append(pair)
        pairs.sort(
            key=lambda item: (
                -(item["influence"] if item["influence"] is not None else -1.0),
                item["field"],
            )
        )
        metric_reports.append(
            {
                "name": name,
                **stats,
                "influence_max": pairs[0]["influence"] if pairs else None,
                "pairs": pairs,
                "skipped_fields": insufficient,
            }
        )

    metric_reports.sort(
        key=lambda item: (
            -(item["influence_max"] if item["influence_max"] is not None else -1.0),
            item["name"],
        )
    )
    return {
        "schema": ANALYSIS_SCHEMA,
        "generated_at": time(),
        "batch_id": batch_id,
        "coverage": {
            "runs_total": len(collected.runs) + len(collected.skipped),
            "runs_analyzed": len(succeeded),
            "excluded": {
                status: count
                for status, count in statuses.items()
                if status != "succeeded"
            },
            "skipped": collected.skipped,
            "constant_fields": constant_fields,
            "metrics_total": len(metric_reports),
        },
        "metrics": metric_reports,
    }
