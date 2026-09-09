"""Immutable, Batch-local completed prefix nodes with atomic publication."""

import pickle
import warnings
from hashlib import sha256
from uuid import uuid4

from .dependencies import matches
from .storage import RecoveryWarning, StorageError, read_record, write_record


def _hex(value, length):
    return (
        isinstance(value, str)
        and len(value) == length
        and all(c in "0123456789abcdef" for c in value)
    )


class PrefixCache:
    def __init__(self, root, signature, seed, serializer):
        self.root = root
        self.key = sha256(pickle.dumps((signature, seed), protocol=4)).hexdigest()
        self.serializer = serializer

    def _directory(self, reference):
        if not isinstance(reference, tuple) or len(reference) != 3:
            raise StorageError("invalid shared-cache reference")
        key, parent, node = reference
        if (
            key != self.key
            or not _hex(key, 64)
            or not _hex(node, 32)
            or not (parent == "root" or _hex(parent, 32))
        ):
            raise StorageError("invalid shared-cache reference")
        directory = self.root / key / parent / node
        if not directory.resolve().is_relative_to(self.root.resolve()):
            raise StorageError("shared-cache reference escaped the Batch directory")
        return directory

    def load(self, reference, position):
        directory = self._directory(reference)
        metadata = read_record(directory / "metadata.pkl", self.serializer)
        if (
            not isinstance(metadata, dict)
            or metadata.get("version") != 1
            or metadata.get("position") != position
            or metadata.get("reference") != reference
        ):
            raise StorageError("invalid shared-cache metadata")
        record = read_record(directory / "snapshot.pkl", self.serializer)
        if (
            not isinstance(record, dict)
            or record.get("position") != position
            or not isinstance(record.get("state"), dict)
            or "output" not in record
            or not isinstance(record.get("name"), str)
            or "rng_state" not in record
        ):
            raise StorageError("invalid shared-cache snapshot")
        return record

    def find(self, parent, position, config):
        directory = self.root / self.key / parent
        for path in sorted(directory.glob("*/metadata.pkl")):
            reference = (self.key, parent, path.parent.name)
            try:
                metadata = read_record(path, self.serializer)
                if metadata.get("position") != position:
                    continue
                if matches(metadata["dependencies"], config):
                    return reference, self.load(reference, position)
            except (
                StorageError,
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
            ) as error:
                warnings.warn(
                    f"ignoring unusable shared cache {path}: {error}",
                    RecoveryWarning,
                    stacklevel=2,
                )
        return None

    def publish(self, parent, record, tracker):
        reference = (self.key, parent, uuid4().hex)
        directory = self._directory(reference)
        # Serializing user output/state can itself read configuration views.
        # Publish dependency metadata only after those reads have been observed.
        write_record(directory / "snapshot.pkl", record, self.serializer)
        write_record(
            directory / "metadata.pkl",
            {
                "version": 1,
                "reference": reference,
                "position": record["position"],
                "dependencies": tracker.export(),
            },
            self.serializer,
        )
        return reference
