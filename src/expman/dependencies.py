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

    def export(self):
        return [
            (path, observation(self.config, path))
            for path in sorted(self.paths, key=repr)
        ]


def matches(dependencies, config):
    return all(observation(config, path) == digest for path, digest in dependencies)
