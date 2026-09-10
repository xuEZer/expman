"""Expand experiment choices and resolve named defaults into independent runs."""

import warnings
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode, SequenceNode


class ConfigError(ValueError):
    """An experiment or defaults file violates the configuration contract."""


class MissingConfigWarning(UserWarning):
    """A named defaults file was absent; explicit parameters are still usable."""


@dataclass(frozen=True)
class _Choice:
    values: tuple[Any, ...]


class _Loader(yaml.SafeLoader):
    def __init__(self, stream, *, allow_choices: bool) -> None:
        super().__init__(stream)
        self.allow_choices = allow_choices

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


def _construct_choice(loader: _Loader, node: SequenceNode) -> _Choice:
    if not loader.allow_choices:
        raise ConstructorError(
            None, None, "!choice is only allowed in experiment files", node.start_mark
        )
    if not isinstance(node, SequenceNode) or not node.value:
        raise ConstructorError(
            None, None, "!choice requires a nonempty sequence", node.start_mark
        )
    return _Choice(tuple(loader.construct_sequence(node, deep=True)))


_Loader.add_constructor("!choice", _construct_choice)


def _check_cycles(value: Any, active: set[int]) -> None:
    if not isinstance(value, (dict, list, _Choice)):
        return
    identity = id(value)
    if identity in active:
        raise ConfigError("recursive YAML aliases are not supported")
    active.add(identity)
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, _Choice):
        children = value.values
    else:
        children = value
    for child in children:
        _check_cycles(child, active)
    active.remove(identity)


def _read_yaml(path: Path, *, allow_choices: bool) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            loader = _Loader(stream, allow_choices=allow_choices)
            try:
                data = loader.get_single_data()
            finally:
                loader.dispose()
        if not isinstance(data, dict):
            raise ConfigError("the document must contain a mapping at its root")
        _check_cycles(data, set())
        return data
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError, ConfigError) as error:
        raise ConfigError(f"{path}: {error}") from error


def _expand(value: Any) -> Iterator[Any]:
    if isinstance(value, _Choice):
        for candidate in value.values:
            yield from _expand(candidate)
    elif isinstance(value, dict):
        for values in product(*(_expand(child) for child in value.values())):
            yield deepcopy(dict(zip(value, values, strict=True)))
    elif isinstance(value, list):
        for values in product(*(_expand(child) for child in value)):
            yield deepcopy(list(values))
    else:
        yield deepcopy(value)


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
        if "name" in value:
            if self.root is None:
                self.root = (_project_root(self.experiment) / "configs").resolve()
            name = value["name"]
            if not all(_safe_segment(segment) for segment in (*keys, name)):
                raise ConfigError(
                    f"{self.experiment}: {location}.name and its field path must use "
                    "nonempty path components without separators, '.' or '..'"
                )
            target = self.root.joinpath(*keys, f"{name}.yaml").resolve()
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
                    self.cache[target] = _read_yaml(target, allow_choices=False)
                except FileNotFoundError:
                    self.cache[target] = None
                    warnings.warn(
                        f"{self.experiment}: no defaults for {location}.name={name!r}: "
                        f"{target}; keeping explicit parameters",
                        MissingConfigWarning,
                        stacklevel=3,
                    )
            defaults = self.cache[target]
            if defaults is not None:
                value = _merge(defaults, value)
                ancestors = (*ancestors, target)
        return {
            key: self.resolve(child, (*keys, key), f"{location}.{key}", ancestors)
            for key, child in value.items()
        }


def load_configs(path: str | Path) -> list[dict[str, Any]]:
    """Load one experiment YAML into independent, fully resolved run dictionaries.

    !choice marks alternatives; independent choices form a Cartesian product in
    YAML field/candidate order. Ordinary lists remain lists. Named nodes load
    defaults from <project root>/configs/<field path>/<name>.yaml. Project roots
    are located by pyproject.toml or .git above the YAML, then the working directory.
    Explicit values win, dictionaries merge recursively, and lists/null replace
    defaults. Defaults cannot contain !choice. Missing defaults warn once per
    resolved file per call. Parsing errors raise ConfigError.

    The result is eager: callers should keep the number of combinations bounded.
    """
    experiment = Path(path).expanduser().resolve()
    try:
        data = _read_yaml(experiment, allow_choices=True)
    except FileNotFoundError as error:
        raise ConfigError(f"experiment file not found: {experiment}") from error
    if "device" in data:
        # A Batch resource list is never an experiment choice.
        from .devices import configured_devices

        configured_devices([data])
    defaults = _Defaults(experiment)
    return [defaults.resolve(run) for run in _expand(data)]
