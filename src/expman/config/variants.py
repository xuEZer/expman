"""Expand combines, grid and scan in order, then deduplicate resolved configurations."""

import json
from collections.abc import Callable, Iterable
from copy import copy, deepcopy
from itertools import product
from typing import Any

import yaml

from expman.config.schema import COMBINES_KEY, GRID_KEY, SCAN_KEY, ConfigError


class _AxisMapping(dict):
    """A whole subtree chosen by an axis supersedes defaults on its parent."""


def _axes(
    tree: Any,
    existing: Any,
    path: tuple[str, ...],
    label: str,
    *,
    known: bool = True,
) -> list[tuple[tuple[str, ...], list]]:
    """Flatten a sparse copy of the data tree into ``(field path, values)`` axes.

    One rule applies at every level: a list is the set of values this node
    takes, a mapping keeps descending. Anything else is a mistake.

    With ``known`` set, every key must already exist in the document, so a typo
    cannot quietly invent a field; ``scan`` declares its own parameters outright
    and turns the check off.
    """
    where = f"{label}.{'.'.join(path)}" if path else label
    if isinstance(tree, list):
        if not tree:
            raise ConfigError(f"{where} needs at least one alternative")
        return [(path, list(tree))]
    if not isinstance(tree, dict):
        raise ConfigError(f"{where} must be a mapping or a list of alternatives")
    if not tree:
        raise ConfigError(f"{where} must not be empty")
    axes: list[tuple[tuple[str, ...], list]] = []
    for key, child in tree.items():
        if known and (not isinstance(existing, dict) or key not in existing):
            raise ConfigError(f"{where}.{key} is not present in the experiment file")
        axes.extend(
            _axes(
                child,
                existing[key] if known else None,
                (*path, key),
                label,
                known=known,
            )
        )
    return axes


def _materialize(document: dict, patches: list[tuple[tuple[str, ...], Any]]) -> dict:
    """Apply ``(path, value)`` patches to a deep copy of the document."""
    result = deepcopy(document)
    for path, value in patches:
        node = result
        for key in path[:-1]:
            child = node.get(key)
            # Detach this branch from any other path pointing at the same YAML alias.
            node[key] = copy(child) if isinstance(child, dict) else {}
            node = node[key]
        node[path[-1]] = deepcopy(
            _AxisMapping(value) if isinstance(value, dict) else value
        )
    return result


def _reject_device_axes(axes: list[tuple[tuple[str, ...], list]]) -> None:
    for path, _ in axes:
        if path and path[0] == "device":
            raise ConfigError(
                "device is the Batch resource list and cannot be an expansion axis"
            )


def _combine_variants(document: dict, candidates: Any) -> list[dict]:
    """Base documents: each candidate merged over the file, independent of the rest."""
    if not candidates:
        raise ConfigError(f"{COMBINES_KEY} must list at least one candidate")
    variants: list[dict] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ConfigError(f"{COMBINES_KEY}[{index}] must be a mapping")
        variants.append(merge(document, candidate))
    return variants


def _grid_variants(document: dict, tree: dict, resolved: Any) -> list[dict]:
    """Every combination of the listed values, in declaration order."""
    axes = _axes(tree, resolved, (), GRID_KEY)
    _reject_device_axes(axes)
    variants: list[dict] = []
    for picks in product(*(values for _, values in axes)):
        patches = [(path, pick) for (path, _), pick in zip(axes, picks, strict=True)]
        variants.append(_materialize(document, patches))
    return variants


def _scan_variants(document: dict, tree: dict) -> list[dict]:
    """A reference point, then one run per remaining value with one axis moved."""
    axes = _axes(tree, None, (), SCAN_KEY, known=False)
    _reject_device_axes(axes)
    # The parameters live in the scan itself. The first value of every axis
    # together is the reference point; each remaining value is tried alone,
    # with the other axes left at theirs.
    reference = [(path, values[0]) for path, values in axes]
    variants = [_materialize(document, reference)]
    for index, (path, values) in enumerate(axes):
        for value in values[1:]:
            patches = list(reference)
            patches[index] = (path, value)
            variants.append(_materialize(document, patches))
    return variants


def deduplicate(variants: Iterable[dict]) -> list[dict]:
    """Drop the configurations that repeat one already produced."""
    unique: list[dict] = []
    seen: set[str] = set()
    for variant in variants:
        try:
            identity = json.dumps(variant, sort_keys=True)
        except TypeError:
            # Dates, binary scalars and sets need their YAML types to remain
            # distinct from ordinary strings after defaults are resolved.
            identity = "yaml:" + yaml.safe_dump(variant, sort_keys=True)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(variant)
    return unique


def merge(defaults: Any, explicit: Any) -> Any:
    if (
        isinstance(defaults, dict)
        and isinstance(explicit, dict)
        and not isinstance(explicit, _AxisMapping)
    ):
        result = deepcopy(defaults)
        for key, value in explicit.items():
            result[key] = (
                merge(defaults[key], value) if key in defaults else deepcopy(value)
            )
        return result
    return deepcopy(explicit)


def expand(
    document: dict,
    combines: Any,
    grid: Any,
    scan: Any,
    resolve_defaults: Callable[[dict], dict],
) -> list[dict]:
    """Nest the control layers outside in: combines, then grid, then scan."""
    bases = [document] if combines is None else _combine_variants(document, combines)
    variants: list[dict] = []
    for base in bases:
        layer = (
            [base]
            if grid is None
            else _grid_variants(base, grid, resolve_defaults(base))
        )
        for item in layer:
            if scan is None:
                variants.append(item)
            else:
                variants.extend(_scan_variants(item, scan))
    return variants
