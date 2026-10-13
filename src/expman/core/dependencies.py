"""User-declared configuration dependencies for stages and snapshots."""

import pickle
from collections.abc import Mapping, Sequence
from hashlib import sha256


def _plain(value):
    """Canonicalize read-only configuration views for stable hashing.

    The same configuration is hashed both as a plain mapping and as a frozen
    view depending on the call site; collapsing both shapes keeps digests equal.
    """
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return frozenset(_plain(item) for item in value)
    return value


def observation(config, path):
    value = config
    try:
        for key in path:
            value = value[key]
    except (KeyError, IndexError, TypeError):
        return None
    return sha256(pickle.dumps(_plain(value), protocol=4)).hexdigest()


def declared_paths(declaration):
    """Resolve a hierarchical declaration into a sorted tuple of paths.

    A declaration mirrors the configuration tree: ``True`` at a node means the
    whole subtree at that position, a mapping recurses into its keys (strings
    for mappings, integers for sequence indices), and ``False`` or ``None``
    means no dependency. The root may itself be ``True`` for the whole
    configuration.
    """
    paths = []
    _collect(declaration, (), paths)
    return tuple(sorted(set(paths), key=repr))


def _collect(node, path, paths):
    if node is True:
        paths.append(path)
        return
    if node is False or node is None:
        return
    if isinstance(node, Mapping):
        for key, child in node.items():
            if isinstance(key, bool) or not isinstance(key, (str, int)):
                raise TypeError(
                    "configuration dependency keys must be strings or integers"
                )
            _collect(child, (*path, key), paths)
        return
    raise TypeError(
        "configuration dependency declaration must be True, False, None or a mapping"
    )


def stage_dependencies(stage, config):
    """Return the declared ``(path, digest)`` dependencies of one Stage class."""
    return [
        (path, observation(config, path))
        for path in declared_paths(stage.config_dependencies(config))
    ]


def matches(dependencies, config):
    return all(observation(config, path) == digest for path, digest in dependencies)


def validate(dependencies):
    if not isinstance(dependencies, list):
        raise ValueError("invalid configuration dependencies")
    for entry in dependencies:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise ValueError("invalid configuration dependency")
        path, digest = entry
        if not isinstance(path, tuple) or any(
            type(key) not in (str, int) for key in path
        ):
            raise ValueError("invalid configuration dependency path")
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("invalid configuration dependency digest")
