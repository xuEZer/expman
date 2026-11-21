"""Workspace-local batch IDs shared by the CLI and dashboard."""

import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

from expman.settings import settings
from expman.storage.base import StorageError, run_lock

REGISTRY_NAME = ".batches.sqlite3"
_DELETING = ".expman-deleting"


def _scan_batches(root: Path) -> list[Path]:
    """Order batch directories by modification time, then by name."""
    found = []
    try:
        for path in root.iterdir():
            if path.name != _DELETING and (path / "batch.pkl").is_file():
                try:
                    found.append((path.stat().st_mtime_ns, path))
                except FileNotFoundError:
                    continue
    except FileNotFoundError:
        return []
    return [path for _, path in sorted(found)]


def _registered(connection: sqlite3.Connection) -> dict[str, int]:
    return dict(connection.execute("SELECT name, id FROM batches"))


def ensure_ids(root: Path | None = None) -> dict[int, Path]:
    """Look up visible batches, registering new names in one write transaction."""
    base = (settings.batches_dir if root is None else Path(root)).resolve()
    candidates = _scan_batches(base)
    if not candidates:
        return {}

    registry = base / REGISTRY_NAME
    known: dict[str, int] = {}
    if registry.exists():
        with closing(sqlite3.connect(f"{registry.as_uri()}?mode=ro", uri=True)) as db:
            # A concurrent creator may have opened the file before committing its schema.
            if db.execute(
                "SELECT 1 FROM sqlite_schema WHERE name='batches'"
            ).fetchone():
                known = _registered(db)

    if any(path.name not in known for path in candidates):
        with closing(sqlite3.connect(registry, timeout=30)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "CREATE TABLE IF NOT EXISTS batches (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE)"
            )
            # Another process may have registered or deleted a batch before we got the lock.
            candidates = _scan_batches(base)
            known = _registered(db)
            occupied = set(known.values())
            next_id = 0
            for path in candidates:
                if path.name in known:
                    continue
                while next_id in occupied:
                    next_id += 1
                db.execute(
                    "INSERT INTO batches (id, name) VALUES (?, ?)", (next_id, path.name)
                )
                known[path.name] = next_id
                next_id += 1
    return {known[path.name]: path for path in candidates}


def _remove_batch_directory(directory: Path) -> None:
    # Keep an interrupted deletion out of discovery. A retry can finish it
    # using the still-reserved registry entry.
    trash = directory.parent / _DELETING
    staged = trash / directory.name
    if trash.is_symlink() or staged.is_symlink() or directory.is_symlink():
        raise StorageError(
            f"refusing to delete a batch through a symbolic link: {directory}"
        )
    if directory.exists():
        if staged.exists():
            raise StorageError(
                f"both {directory} and its pending deletion {staged} exist"
            )
        if not (directory / "batch.pkl").is_file():
            raise StorageError(
                f"refusing to remove a directory without a batch manifest: {directory}"
            )
        with run_lock(directory):
            trash.mkdir(mode=0o700, exist_ok=True)
            directory.rename(staged)
    if staged.exists():
        shutil.rmtree(staged)
        if not any(trash.iterdir()):
            trash.rmdir()


def delete_batch(root: Path, batch_id: int) -> Path:
    """Delete the registered batch and release its ID after its files are gone."""
    base = root.resolve()
    registry = base / REGISTRY_NAME
    if registry.is_symlink():
        raise StorageError(
            f"refusing to delete through a linked batch registry: {registry}"
        )
    if not registry.is_file():
        raise StorageError(f"unknown batch ID {batch_id} under {root}")
    with (
        closing(
            sqlite3.connect(f"{registry.as_uri()}?mode=rw", uri=True, timeout=30)
        ) as db,
        db,
    ):
        row = db.execute(
            "SELECT name FROM batches WHERE id = ?", (batch_id,)
        ).fetchone()
        if row is None:
            raise StorageError(f"unknown batch ID {batch_id} under {root}")
        name = row[0]
        if (
            not isinstance(name, str)
            or not name
            or name in (".", "..", _DELETING, REGISTRY_NAME)
            or Path(name).name != name
        ):
            raise StorageError(f"invalid registered batch directory: {name!r}")
        directory = base / name
        db.execute("BEGIN IMMEDIATE")
        # Pin the name seen by this request if another writer released and reused the ID.
        deleted = db.execute(
            "DELETE FROM batches WHERE id = ? AND name = ?", (batch_id, name)
        )
        if deleted.rowcount != 1:
            raise StorageError(
                f"batch ID {batch_id} changed during deletion under {root}"
            )
        _remove_batch_directory(directory)
    return directory
