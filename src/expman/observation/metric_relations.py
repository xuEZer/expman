"""Identify controlled contrasts in observed batch metrics."""

import json


def _coordinates(variables, records):
    """Encode each config value by its first occurrence in the expanded YAML."""
    levels = [{} for _ in variables]
    points = []
    for row in records:
        point = []
        for variable, values in zip(variables, levels, strict=True):
            key = variable["id"]
            # JSON keeps scalar types distinct. A missing field gets its own
            # key, so it remains distinct from an explicit null.
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


def compare_metrics(
    variables: list[dict], metrics: list[dict], records: list[dict]
) -> tuple[list[tuple[int, ...]], list[list[int]], list[list[int]]]:
    """Return each row's coordinates and each metric's dependent and unresolved axes.

    An axis is an observed dependency when two measured values differ at
    different levels of that axis with every other configuration value fixed.
    """
    row_points, level_counts = _coordinates(variables, records)
    points, measured = _observations(row_points, metrics, records)
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
    return row_points, dependencies, undetermined
