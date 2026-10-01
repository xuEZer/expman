"""Expand experiment variations and resolve named defaults into independent runs.

An experiment file is a mapping. Three optional control keys shape it, and they
nest from the outside in:

``combines``
    Turns the collection into whole candidates. Its value is a **list of
    mappings**; each one is a sparse overlay merged over the file to give one
    base document. Candidates are tried in order and never mixed with each
    other, so it pairs fields that move together.

``grid``
    Turns each base document into a Cartesian grid. Its value is a sparse copy
    of the data tree in which a **list** means the node takes one of these
    values and a **mapping** keeps descending. The runs are the product of the
    axes.

``scan``
    Turns each base document into experiments tried one axis at a time. Its
    value is a sparse tree that declares its own parameters: every axis starts
    at its first listed value, and each remaining value is tried alone with the
    other axes left at theirs.

Combining them multiplies the layers -- every candidate is gridded, and every
grid point is scanned. ``defaults`` appears on any mapping node and names a
file under ``<project root>/configs/`` to merge in as defaults. It is a lookup
directive and never reaches the resolved run.

``load_configs`` returns one dictionary per Run. The result is eager: callers
should keep the number of combinations bounded.
"""

import json
import warnings
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

COMBINES_KEY = "combines"
GRID_KEY = "grid"
SCAN_KEY = "scan"
DEFAULTS_KEY = "defaults"

_CONTROL_KEYS = (COMBINES_KEY, GRID_KEY, SCAN_KEY)


class ConfigError(ValueError):
    """An experiment or defaults file violates the configuration contract."""


class MissingConfigWarning(UserWarning):
    """A named defaults file was absent; explicit parameters are still usable."""


class _Loader(yaml.SafeLoader):
    """A SafeLoader that rejects merge keys, non-string keys and duplicate keys."""

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict:
        result = {}
        for key_node, value_node in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                raise ConstructorError(
                    None, None, "YAML merge keys are not supported", key_node.start_mark
                )
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConstructorError(
                    None,
                    None,
                    "configuration keys must be strings",
                    key_node.start_mark,
                )
            if key in result:
                raise ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark
                )
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def _check_cycles(value: Any, active: set[int]) -> None:
    if not isinstance(value, (dict, list)):
        return
    identity = id(value)
    if identity in active:
        raise ConfigError("recursive YAML aliases are not supported")
    active.add(identity)
    children = value.values() if isinstance(value, dict) else value
    for child in children:
        _check_cycles(child, active)
    active.remove(identity)


def _read_yaml(path: Path, *, allow_control: bool) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            loader = _Loader(stream)
            try:
                data = loader.get_single_data()
            finally:
                loader.dispose()
        if not isinstance(data, dict):
            raise ConfigError("the document must be a mapping at its root")
        if not allow_control:
            for key in _CONTROL_KEYS:
                if key in data:
                    raise ConfigError(f"{key} is only allowed in experiment files")
        _check_cycles(data, set())
        return data
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError, ConfigError) as error:
        raise ConfigError(f"{path}: {error}") from error


def _identity(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


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
            if not isinstance(child, dict):
                child = {}
                node[key] = child
            node = child
        node[path[-1]] = deepcopy(value)
    return result


def _reject_device_axes(axes: list[tuple[tuple[str, ...], list]]) -> None:
    for path, _ in axes:
        if path and path[0] == "device":
            raise ConfigError(
                "device is the Batch resource list and cannot be an expansion axis"
            )


def _combine_variants(document: dict, candidates: Any) -> list[dict]:
    """Base documents: each candidate merged over the file, independent of the rest."""
    if not isinstance(candidates, list):
        raise ConfigError(f"{COMBINES_KEY} must be a list of candidate mappings")
    if not candidates:
        raise ConfigError(f"{COMBINES_KEY} must list at least one candidate")
    variants: list[dict] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ConfigError(f"{COMBINES_KEY}[{index}] must be a mapping")
        variants.append(_merge(document, candidate))
    return variants


def _grid_variants(document: dict, tree: dict, resolved: Any) -> list[dict]:
    """Every combination of the listed values, in declaration order."""
    axes = _axes(tree, resolved, (), GRID_KEY)
    _reject_device_axes(axes)
    variants: list[dict] = []
    for picks in product(*(values for _, values in axes)):
        patches = [(axes[index][0], picks[index]) for index in range(len(axes))]
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


def _deduplicate(variants: list[dict]) -> list[dict]:
    """Drop the configurations that repeat one already produced."""
    unique: list[dict] = []
    seen: set[str] = set()
    for variant in variants:
        identity = _identity(variant)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(variant)
    return unique


def _merge(defaults: Any, explicit: Any) -> Any:
    if isinstance(defaults, dict) and isinstance(explicit, dict):
        result = deepcopy(defaults)
        for key, value in explicit.items():
            result[key] = (
                _merge(defaults[key], value) if key in defaults else deepcopy(value)
            )
        return result
    return deepcopy(explicit)


def _safe_segment(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value not in (".", "..")
        and not any(character in value for character in ("/", "\\", "\0"))
    )


def _project_root(experiment: Path) -> Path:
    for start in (experiment.parent, Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").is_file() or (
                candidate / ".git"
            ).exists():
                return candidate
    raise ConfigError(
        f"{experiment}: cannot locate project root (pyproject.toml or .git)"
    )


class _Defaults:
    def __init__(self, experiment: Path) -> None:
        self.experiment = experiment
        self.root: Path | None = None
        # A cached entry is what _read_yaml returned, or None when the file is
        # missing.
        self.cache: dict[Path, dict[str, Any] | None] = {}

    def resolve(
        self,
        value: Any,
        keys: tuple[str, ...] = (),
        location: str = "$",
        ancestors: tuple[Path, ...] = (),
    ) -> Any:
        if isinstance(value, list):
            # Lists preserve their structure; indices are not directory names.
            return [
                self.resolve(child, keys, f"{location}[{index}]", ancestors)
                for index, child in enumerate(value)
            ]
        if not isinstance(value, dict):
            return deepcopy(value)
        source = value
        if DEFAULTS_KEY in value:
            source = {key: child for key, child in value.items() if key != DEFAULTS_KEY}
            defaults, target = self._lookup(
                value[DEFAULTS_KEY], keys, location, ancestors
            )
            if defaults is not None:
                source = _merge(defaults, source)
                ancestors = (*ancestors, target)
        return {
            key: self.resolve(child, (*keys, key), f"{location}.{key}", ancestors)
            for key, child in source.items()
        }

    def _lookup(
        self,
        lookup: Any,
        keys: tuple[str, ...],
        location: str,
        ancestors: tuple[Path, ...],
    ) -> tuple[dict[str, Any] | None, Path]:
        if not _safe_segment(lookup):
            raise ConfigError(
                f"{self.experiment}: {location}.{DEFAULTS_KEY} must be a nonempty "
                "name without separators, '.' or '..'"
            )
        if self.root is None:
            self.root = (_project_root(self.experiment) / "configs").resolve()
        if not all(_safe_segment(segment) for segment in keys):
            raise ConfigError(
                f"{self.experiment}: {location} and its field path must use "
                "nonempty path components without separators, '.' or '..'"
            )
        target = self.root.joinpath(*keys, f"{lookup}.yaml").resolve()
        if not target.is_relative_to(self.root):
            raise ConfigError(
                f"{self.experiment}: {location} resolves outside {self.root}"
            )
        if target in ancestors:
            raise ConfigError(
                f"{self.experiment}: cyclic defaults reference at {target}"
            )
        if target not in self.cache:
            try:
                self.cache[target] = _read_yaml(target, allow_control=False)
            except FileNotFoundError:
                self.cache[target] = None
                warnings.warn(
                    f"{self.experiment}: no defaults for {location}.{DEFAULTS_KEY}"
                    f"={lookup!r}: {target}; keeping explicit parameters",
                    MissingConfigWarning,
                    stacklevel=3,
                )
        return self.cache[target], target


def _expand(
    document: dict, combines: Any, grid: Any, scan: Any, defaults: _Defaults
) -> list[dict]:
    """Nest the control layers outside in: combines, then grid, then scan."""
    bases = [document] if combines is None else _combine_variants(document, combines)
    variants: list[dict] = []
    for base in bases:
        layer = (
            [base]
            if grid is None
            else _grid_variants(base, grid, defaults.resolve(base))
        )
        for item in layer:
            if scan is None:
                variants.append(item)
            else:
                variants.extend(_scan_variants(item, scan))
    return variants


def load_configs(path: str | Path) -> list[dict[str, Any]]:
    """Load one experiment YAML into independent, fully resolved run dictionaries.

    A top-level ``combines``, ``grid`` or ``scan`` section turns the document
    into a collection. They nest from the outside in::

        device: [0]
        seed: 42

        combines:                    # whole candidates: 2 runs
          - {dataset: {name: A}, model: {name: saits}}
          - {dataset: {name: B}, model: {name: brits}}

        grid:                        # every combination: 2 x 2 = 4 runs
          seed: [0, 1]
          model: {rank: [2, 4]}

        scan:                        # one axis at a time: 3 runs
          model:
            rank: [2, 4]             # (2, 1e-05), (4, 1e-05), (2, 0.001)
            epsilon: [0.00001, 0.001]

    ``combines`` takes a list of mappings merged over the file, one run each,
    and pairs fields that move together. ``grid`` returns the product of its
    axes in declaration order, the last axis moving fastest, and every path it
    lists must already exist in the document. ``scan`` declares its own
    parameters -- they need not appear in the file, and a value written there is
    replaced -- and returns the combination of their first values as the
    reference point, then one run per remaining value with a single axis moved.

    Writing several of them multiplies the layers: every candidate is gridded,
    and every grid point is scanned, so the innermost layer varies fastest. A
    run whose configuration repeats one already produced once defaults are
    resolved is not produced again.

    Any mapping node may carry ``defaults: <name>``, which merges
    ``<project root>/configs/<field path>/<name>.yaml`` underneath it. Explicit
    values win, dictionaries merge recursively, and lists/null replace defaults.
    Project roots are located by pyproject.toml or .git above the YAML, then the
    working directory. Defaults files cannot contain ``combines``, ``grid`` or
    ``scan``. Missing defaults warn once per resolved file per call. Parsing
    errors raise ConfigError.
    """
    experiment = Path(path).expanduser().resolve()
    try:
        document = _read_yaml(experiment, allow_control=True)
    except FileNotFoundError as error:
        raise ConfigError(f"experiment file not found: {experiment}") from error

    combines = document.pop(COMBINES_KEY, None)
    grid = document.pop(GRID_KEY, None)
    scan = document.pop(SCAN_KEY, None)
    if grid is not None and not isinstance(grid, dict):
        raise ConfigError(f"{experiment}: {GRID_KEY} must be a mapping of axes")
    if scan is not None and not isinstance(scan, dict):
        raise ConfigError(f"{experiment}: {SCAN_KEY} must be a mapping of axes")

    defaults = _Defaults(experiment)
    variants = _expand(document, combines, grid, scan, defaults)

    device_configs = [
        variant
        for variant in variants
        if isinstance(variant, dict) and "device" in variant
    ]
    if device_configs:
        # A device list is a Batch resource, never an experiment axis.
        from .devices import configured_devices

        configured_devices(device_configs)

    return _deduplicate([defaults.resolve(variant) for variant in variants])
