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


def _read_yaml(path: Path, *, allow_choices: bool) -> dict[str, Any] | _Choice:
    try:
        with path.open(encoding="utf-8") as stream:
            loader = _Loader(stream, allow_choices=allow_choices)
            try:
                data = loader.get_single_data()
            finally:
                loader.dispose()
        if not isinstance(data, (dict, _Choice)):
            raise ConfigError(
                "the document must be a mapping or a !choice of mappings at its root"
            )
        if isinstance(data, dict) and not allow_choices and "sweep" in data:
            raise ConfigError("sweep is only allowed in experiment files")
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


_SWEEP_KEYS = frozenset({"axes", "mode", "include_baseline"})


def _axis_parts(text: str) -> tuple[str, ...]:
    if not isinstance(text, str) or not text:
        raise ConfigError("sweep axis names must be nonempty dotted paths")
    parts = tuple(text.split("."))
    if any(not part for part in parts):
        raise ConfigError(f"invalid sweep axis {text!r}")
    return parts


def _axis_path(config: Any, text: str) -> tuple[str, ...]:
    """Resolve an axis name, requiring the path to exist in the experiment file."""
    parts = _axis_parts(text)
    node = config
    for component in parts:
        if isinstance(node, _Choice):
            raise ConfigError(
                f"sweep axis {text!r} crosses a !choice; pin the choice first"
            )
        if not isinstance(node, dict) or component not in node:
            raise ConfigError(
                f"sweep axis {text!r} is not present in the experiment file"
            )
        node = node[component]
    if isinstance(node, _Choice):
        raise ConfigError(f"sweep axis {text!r} names a !choice; pin it first")
    return parts


def _set_axis(config: dict, parts: tuple[str, ...], value: Any) -> None:
    node = config
    for component in parts[:-1]:
        node = node[component]
    node[parts[-1]] = deepcopy(value)


def _sweep_variants(config: dict, spec: Any) -> list[dict]:
    """Expand a top-level ``sweep`` section into one mapping per run.

    ``ofat`` (the default) varies one axis at a time from the file's own values,
    so the run count is the sum of the axis sizes instead of their product;
    ``grid`` takes the Cartesian product of the axes. ``include_baseline``
    (default true) prepends the unmodified experiment file as the control run.
    """
    if not isinstance(spec, dict):
        raise ConfigError("sweep must be a mapping")
    unknown = sorted(set(spec) - _SWEEP_KEYS)
    if unknown:
        raise ConfigError(f"unknown sweep keys: {unknown}")
    axes = spec.get("axes")
    if not isinstance(axes, dict) or not axes:
        raise ConfigError("sweep.axes must be a nonempty mapping")
    mode = spec.get("mode", "ofat")
    if mode not in ("ofat", "grid"):
        raise ConfigError("sweep.mode must be 'ofat' or 'grid'")
    include_baseline = spec.get("include_baseline", True)
    if not isinstance(include_baseline, bool):
        raise ConfigError("sweep.include_baseline must be a boolean")

    resolved = []
    for text, values in axes.items():
        if (
            isinstance(values, _Choice)
            or not isinstance(values, (list, tuple))
            or not values
        ):
            raise ConfigError(f"sweep axis {text!r} must be a nonempty list")
        resolved.append((_axis_path(config, text), values))

    variants = [deepcopy(config)] if include_baseline else []
    if mode == "ofat":
        for parts, values in resolved:
            for value in values:
                variant = deepcopy(config)
                _set_axis(variant, parts, value)
                variants.append(variant)
    else:
        for combination in product(*(values for _, values in resolved)):
            variant = deepcopy(config)
            for (parts, _), value in zip(resolved, combination, strict=True):
                _set_axis(variant, parts, value)
            variants.append(variant)
    return variants


def _device_configs(value: Any) -> list[dict]:
    """Collect the raw configs that declare ``device`` for early validation."""
    if isinstance(value, _Choice):
        configs = []
        for candidate in value.values:
            configs.extend(_device_configs(candidate))
        return configs
    if isinstance(value, dict) and "device" in value:
        return [value]
    return []


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
    defaults. Defaults cannot contain !choice or sweep. Missing defaults warn once
    per resolved file per call. Parsing errors raise ConfigError.

    A top-level ``sweep`` mapping turns the file into a sensitivity scan:

        sweep:
          mode: ofat            # or grid
          include_baseline: true
          axes:
            tip.rank: [2, 4]

    The file's own fields are the baseline; each axis is a dotted path that must
    already exist in the experiment file. ``ofat`` varies one axis at a time, so
    b0/c0 stay pinned while a is scanned; ``grid`` takes the axis product.

    The document root may also be a ``!choice`` of complete mappings, which pairs
    sibling fields (dataset A with model a, dataset B with model b) instead of
    taking the product of independent fields. Candidates may contain further
    ``!choice`` nodes. A root ``!choice`` and ``sweep`` cannot be combined.

    The result is eager: callers should keep the number of combinations bounded.
    """
    experiment = Path(path).expanduser().resolve()
    try:
        data = _read_yaml(experiment, allow_choices=True)
    except FileNotFoundError as error:
        raise ConfigError(f"experiment file not found: {experiment}") from error
    root_choice = isinstance(data, _Choice)
    if root_choice:
        variants = [data]
    else:
        has_sweep = "sweep" in data
        spec = data.pop("sweep", None)
        variants = _sweep_variants(data, spec) if has_sweep else [data]
    device_configs = []
    for variant in variants:
        device_configs.extend(_device_configs(variant))
    if device_configs:
        # A Batch resource list is never an experiment choice or a sweep axis.
        from .devices import configured_devices

        configured_devices(device_configs)
    defaults = _Defaults(experiment)
    runs = []
    for variant in variants:
        for run in _expand(variant):
            if not isinstance(run, dict):
                raise ConfigError(
                    f"{experiment}: every root !choice candidate must be a mapping"
                )
            if root_choice and "sweep" in run:
                raise ConfigError(
                    f"{experiment}: sweep is a top-level section and cannot appear "
                    "inside a root !choice candidate"
                )
            runs.append(defaults.resolve(run))
    return runs
