"""Resolve named defaults and expand experiment YAML files into one run list."""

import warnings
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from expman.config.schema import (
    COMBINES_KEY,
    CONFIGS_DIR,
    CONTROL_KEYS,
    DEFAULTS_KEY,
    GRID_KEY,
    SCAN_KEY,
    ConfigError,
    _safe_segment,
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


class _Defaults:
    def __init__(
        self, experiment: Path, cache: dict[Path, dict[str, Any] | None]
    ) -> None:
        self.experiment = experiment
        self.root = CONFIGS_DIR.resolve()
        # All experiments in a batch share defaults. None records a missing file.
        self.cache = cache

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


def _expand_experiment(
    path: str | Path, defaults_cache: dict[Path, dict[str, Any] | None]
) -> Iterator[dict[str, Any]]:
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

    defaults = _Defaults(experiment, defaults_cache)
    combines = document.pop(COMBINES_KEY, None)
    grid = document.pop(GRID_KEY, None)
    scan = document.pop(SCAN_KEY, None)
    for variant in expand(document, combines, grid, scan, defaults.resolve):
        yield defaults.resolve(variant)


def _load_configs(path: str | Path, *more_paths: str | Path) -> list[dict[str, Any]]:
    """Expand each experiment independently and keep the first occurrence of each config."""
    defaults_cache: dict[Path, dict[str, Any] | None] = {}
    configs = deduplicate(
        config
        for source in (path, *more_paths)
        for config in _expand_experiment(source, defaults_cache)
    )
    if any("device" in cfg for cfg in configs):
        configured_devices(configs)
    return configs
