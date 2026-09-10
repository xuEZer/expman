"""Conservative configuration read dependencies for completed stages."""

import pickle
from copy import deepcopy
from hashlib import sha256


def observation(config, path):
    value = config
    try:
        for key in path:
            value = value[key]
    except (KeyError, IndexError, TypeError):
        return None
    return sha256(pickle.dumps(value, protocol=4)).hexdigest()


class ConfigurationReads:
    def __init__(self, config):
        self.config = deepcopy(config)
        self.active = False
        self.paths = set()

    def begin(self):
        self.paths = set()
        self.active = True

    def end(self):
        self.active = False

    def record(self, path):
        if self.active:
            try:
                self.paths.add(tuple(path))
            except TypeError:
                self.paths.add(())

    def restore(self, dependencies):
        if dependencies is None:
            self.record(())
            return
        validate(dependencies)
        if not matches(dependencies, self.config):
            raise ValueError("checkpoint configuration dependencies changed")
        for path, _ in dependencies:
            self.record(path)

    def export(self):
        return [
            (path, observation(self.config, path))
            for path in sorted(self.paths, key=repr)
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
