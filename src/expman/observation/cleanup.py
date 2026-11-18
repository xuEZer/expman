"""Reclaim unreferenced shared snapshots in an idle batch directory."""

import shutil
from dataclasses import dataclass
from pathlib import Path

from expman.storage.base import (
    PickleSerializer,
    Serializer,
    StorageError,
    read_record,
    run_lock,
)


@dataclass
class CleanReport:
    files: int = 0
    bytes: int = 0

    @property
    def empty(self) -> bool:
        return self.files == 0

    def summary(self) -> str:
        return f"{self.files} 个文件 / {self.bytes / 1e6:.1f} MB"


def _references(root: Path, serializer: Serializer) -> set[tuple]:
    references = set()
    for experiment in (root / "experiments").iterdir():
        stages = experiment / "stages"
        if not stages.is_dir():
            continue
        # Prefix reuse is only enabled for the top-level pipeline.
        for directory in stages.iterdir():
            if not directory.is_dir():
                continue
            path = directory / "completed.pkl"
            if not path.is_file():
                continue
            record = read_record(path, serializer)
            if not isinstance(record, dict):
                raise StorageError(f"invalid completed stage record: {path}")
            if "cache_ref" in record:
                ref = record["cache_ref"]
                if (
                    not isinstance(ref, tuple)
                    or len(ref) != 3
                    or not all(isinstance(part, str) for part in ref)
                ):
                    raise StorageError(f"invalid shared-cache reference: {path}")
                references.add(ref)
    return references


def clean(
    output_dir: str | Path, *, serializer: Serializer | None = None
) -> CleanReport:
    """Delete unreferenced cache nodes, preserving resumability and lazy results.

    Use the batch's serializer when it differs from the default. Unreadable
    top-level completion records abort the scan before any nodes are removed.
    """
    root = Path(output_dir)
    cache = root / "cache"
    report = CleanReport()
    if not cache.is_dir():
        return report
    with run_lock(root):
        referenced = _references(
            root, serializer if serializer is not None else PickleSerializer()
        )
        for node in cache.glob("*/*/*"):
            parts = node.relative_to(cache).parts
            if parts in referenced or not node.is_dir():
                continue
            if node.resolve() != root.resolve() / "cache" / Path(*parts):
                raise StorageError(f"unexpected link in shared cache: {node}")
            sizes = [item.stat().st_size for item in node.iterdir() if item.is_file()]
            shutil.rmtree(node)
            report.files += len(sizes)
            report.bytes += sum(sizes)
    return report
