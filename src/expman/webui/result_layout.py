"""Arrange the observed configurations and metrics into a result table."""

import json
from collections import Counter


def _coordinates(variables, records):
    """Encode each config value by its first occurrence in the expanded YAML."""
    levels = [{} for _ in variables]
    points = []
    for row in records:
        point = []
        for variable, values in zip(variables, levels, strict=True):
            key = variable["id"]
            # Missing and explicit null are different settings. JSON also keeps
            # bools distinct from integers and normalizes object key order.
            value = json.dumps(row[key], sort_keys=True) if key in row else None
            point.append(values.setdefault(value, len(values)))
        points.append(tuple(point))
    return points, [len(values) for values in levels]


def _observations(points, metrics, records):
    """Collect sparse measurements by metric and distinct configuration."""
    point_ids = {}
    measured = {metric["id"]: {} for metric in metrics}
    for point, row in zip(points, records, strict=True):
        point_id = point_ids.setdefault(point, len(point_ids))
        for key, value in row.items():
            if key not in measured or value is None:
                continue
            samples = measured[key]
            if samples.setdefault(point_id, value) != value:
                # Conflicting repetitions cannot establish a controlled contrast.
                samples[point_id] = None
    return list(point_ids), list(measured.values())


def _relations(points, level_counts, measured):
    """Classify each axis by comparing runs with matching other settings."""
    dependencies = [[] for _ in measured]
    undetermined = [[] for _ in measured]
    complete = [
        len(values) == len(points) and None not in values.values()
        for values in measured
    ]
    for axis, levels in enumerate(level_counts):
        # Reuse the same control groups for all metric columns.
        contexts = {}
        groups = []
        for point in points:
            context = point[:axis] + point[axis + 1 :]
            groups.append(contexts.setdefault(context, len(contexts)))
        crossed = levels >= 2 and len(points) == levels * len(contexts)
        for metric, values in enumerate(measured):
            first = {}
            for point_id, value in values.items():
                if (
                    value is not None
                    and first.setdefault(groups[point_id], value) != value
                ):
                    dependencies[metric].append(axis)
                    break
            else:
                # Exclude the axis only when every observed context has all
                # its levels measured. Sparse scans can leave effects unresolved.
                if not crossed or not complete[metric]:
                    undetermined[metric].append(axis)
    return dependencies, undetermined


def _axis_order(count, dependencies):
    """Favor axes that bring the most metric columns close to being ready."""
    remaining = list(range(count))
    ordered = []
    pending = Counter(frozenset(axes) for axes in dependencies if axes)
    while pending:
        scores = dict.fromkeys(remaining, 0.0)
        for axes, frequency in pending.items():
            for axis in axes:
                scores[axis] += frequency / len(axes)
        chosen = max(remaining, key=scores.__getitem__)
        ordered.append(chosen)
        remaining.remove(chosen)
        reduced = Counter()
        for axes, frequency in pending.items():
            still_needed = axes - {chosen}
            if still_needed:
                reduced[still_needed] += frequency
        pending = reduced
    return ordered + remaining


def layout_results(
    variables: list[dict], metrics: list[dict], records: list[dict]
) -> dict:
    """Order result columns and rows, retaining a row for every configuration.

    Inputs use distinct column ids and sparse row dictionaries. Metric values
    are numeric. Place each metric after its last observed dependency and
    any unresolved axes immediately following it. With no observed dependency,
    place the metric after any leading unresolved axes.
    """
    points, level_counts = _coordinates(variables, records)
    settings, measured = _observations(points, metrics, records)
    dependencies, undetermined = _relations(settings, level_counts, measured)
    axes = _axis_order(len(variables), dependencies)
    positions = {axis: position for position, axis in enumerate(axes)}
    slots = [[] for _ in range(len(axes) + 1)]
    for metric, depends, unknown in zip(
        metrics, dependencies, undetermined, strict=True
    ):
        position = 1 + max((positions[axis] for axis in depends), default=-1)
        unresolved = set(unknown)
        while position < len(axes) and axes[position] in unresolved:
            position += 1
        slots[position].append(
            {
                **metric,
                "kind": "metric",
                "depends_on": [variables[axis]["id"] for axis in depends],
                "undetermined": [variables[axis]["id"] for axis in unknown],
            }
        )
    columns = list(slots[0])
    for position, axis in enumerate(axes):
        columns.append({**variables[axis], "kind": "config"})
        columns.extend(slots[position + 1])
    rows = sorted(
        range(len(records)), key=lambda i: tuple(points[i][axis] for axis in axes)
    )
    return {"columns": columns, "rows": [records[index] for index in rows]}
