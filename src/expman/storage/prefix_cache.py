"""Immutable, Batch-local completed prefix nodes with atomic publication."""

import pickle
import warnings
from hashlib import sha256
from uuid import uuid4

from expman.storage.base import (
    RecoveryWarning,
    StorageError,
    read_record,
    write_record,
)
from expman.storage.run import stage_seconds, validate_snapshot


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

    def load(self, reference, position) -> dict:
        directory = self._directory(reference)
        self.metadata(reference, position)
        record = read_record(directory / "snapshot.pkl", self.serializer)
        validate_snapshot(record, position, shared=True)
        return record

    def _candidates(self, parent, position, dependencies):
        """Yield the usable ``(reference, metadata)`` nodes for one position."""
        directory = self.root / self.key / parent
        for path in sorted(directory.glob("*/metadata.pkl")):
            reference = (self.key, parent, path.parent.name)
            try:
                metadata = self.metadata(reference, position)
                if metadata["dependencies"] == dependencies:
                    yield reference, metadata
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
                    stacklevel=3,
                )

    def metadata(self, reference, position) -> dict:
        """Read the dependencies, duration and metrics of a published node.

        The scheduler uses these fields to reuse a Stage without loading the
        output into the parent process.
        """
        directory = self._directory(reference)
        metadata = read_record(directory / "metadata.pkl", self.serializer)
        if (
            not isinstance(metadata, dict)
            or metadata.get("version") != 1
            or metadata.get("position") != position
            or metadata.get("reference") != reference
            or not isinstance(metadata.get("dependencies"), list)
            or not isinstance(metadata.get("metrics"), list)
        ):
            raise StorageError("invalid shared-cache metadata")
        stage_seconds(metadata)
        return metadata

    def find(
        self, parent, position, dependencies
    ) -> tuple[tuple[str, str, str], dict] | None:
        for reference, _metadata in self._candidates(parent, position, dependencies):
            try:
                return reference, self.load(reference, position)
            except (
                StorageError,
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
            ) as error:
                warnings.warn(
                    f"ignoring unusable shared cache {reference}: {error}",
                    RecoveryWarning,
                    stacklevel=2,
                )
        return None

    def find_metadata(
        self, parent, position, dependencies
    ) -> tuple[tuple[str, str, str], dict] | None:
        """Like ``find``, but returns the metadata without the snapshot."""
        for reference, metadata in self._candidates(parent, position, dependencies):
            return reference, metadata
        return None

    def publish(self, parent, record) -> tuple[str, str, str]:
        reference = (self.key, parent, uuid4().hex)
        directory = self._directory(reference)
        write_record(directory / "snapshot.pkl", record, self.serializer)
        write_record(
            directory / "metadata.pkl",
            {
                "version": 1,
                "reference": reference,
                "position": record["position"],
                "dependencies": record["config_dependencies"],
                # Everything a reusing scheduler needs, so that it never has to
                # read the snapshot itself.
                "elapsed_seconds": record["elapsed_seconds"],
                "metrics": record["metrics"],
            },
            self.serializer,
        )
        return reference
