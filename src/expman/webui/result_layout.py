"""Arrange the observed configurations and metrics into a result table."""

from collections import Counter

from expman.observation.metric_relations import compare_metrics


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
    points, dependencies, undetermined = compare_metrics(variables, metrics, records)
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
