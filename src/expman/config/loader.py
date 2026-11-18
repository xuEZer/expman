"""Read experiment YAML, resolve named defaults and expand the run list."""

import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from expman.config.schema import (
    COMBINES_KEY,
    CONTROL_KEYS,
    DEFAULTS_KEY,
    GRID_KEY,
    SCAN_KEY,
    ConfigError,
    configured_devices,
)
from expman.config.variants import deduplicate, expand, merge


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
            for key in CONTROL_KEYS:
                if key in data:
                    raise ConfigError(f"{key} is only allowed in experiment files")
        _check_cycles(data, set())
        return data
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError, ConfigError) as error:
        raise ConfigError(f"{path}: {error}") from error


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
        while DEFAULTS_KEY in source:
            explicit = {
                key: child for key, child in source.items() if key != DEFAULTS_KEY
            }
            defaults, target = self._lookup(
                source[DEFAULTS_KEY], keys, location, ancestors
            )
            if defaults is None:
                source = explicit
                break
            source = merge(defaults, explicit)
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


def load_configs(path: str | Path) -> list[dict[str, Any]]:
    """Load one experiment YAML into independent, fully resolved run dictionaries.

    A top-level ``combines``, ``grid`` or ``scan`` section turns the document
    into a collection. They nest from the outside in::

        device: [0]
        seed: 42
        model: {width: 64}

        combines:                    # whole candidates: 2 runs
          - {dataset: {name: A}, model: {name: saits}}
          - {dataset: {name: B}, model: {name: brits}}

        grid:                        # every combination: 2 x 2 = 4 runs
          seed: [0, 1]
          model: {width: [64, 128]}

        scan:                        # one axis at a time: 3 runs
          model:
            rank: [2, 4]             # (2, 1e-05), (4, 1e-05), (2, 0.001)
            epsilon: [0.00001, 0.001]

    ``combines`` takes a list of mappings merged over the file, one run each,
    and pairs fields that move together. ``grid`` returns the product of its
    axes in declaration order, the last axis moving fastest, and every path it
    lists must exist after merging the candidate's defaults. ``scan`` declares
    its own parameters. Their first values form the reference point. Each
    remaining value changes one axis while the others stay at the reference.

    Writing several of them multiplies the layers: every candidate is gridded,
    and every grid point is scanned, so the innermost layer varies fastest. A
    run whose configuration repeats one already produced once defaults are
    resolved is not produced again. The example yields 24 distinct runs.

    Any mapping node may carry ``defaults: <name>``, which merges
    ``<project root>/configs/<field path>/<name>.yaml`` underneath it. Explicit
    values win, dictionaries merge recursively, and lists/null replace defaults.
    Project roots are located by pyproject.toml or .git above the YAML, then the
    working directory. Defaults files cannot declare top-level ``combines``,
    ``grid`` or ``scan``. Missing defaults warn once per resolved file per call.
    Parsing errors raise ConfigError.
    """
    experiment = Path(path).expanduser().resolve()
    try:
        document = _read_yaml(experiment, allow_control=True)
    except FileNotFoundError as error:
        raise ConfigError(f"experiment file not found: {experiment}") from error

    for key, expected, description in (
        (COMBINES_KEY, list, "a list of candidate mappings"),
        (GRID_KEY, dict, "a mapping of axes"),
        (SCAN_KEY, dict, "a mapping of axes"),
    ):
        if key in document and not isinstance(document[key], expected):
            raise ConfigError(f"{experiment}: {key} must be {description}")

    defaults = _Defaults(experiment)
    combines = document.pop(COMBINES_KEY, None)
    grid = document.pop(GRID_KEY, None)
    scan = document.pop(SCAN_KEY, None)
    variants = expand(document, combines, grid, scan, defaults.resolve)
    configs = [defaults.resolve(variant) for variant in variants]
    if any("device" in cfg for cfg in configs):
        configured_devices(configs)
    return deduplicate(configs)
