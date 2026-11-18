"""Stable workspace-local batch IDs shared by the CLI and dashboard."""

import sqlite3
from contextlib import closing
from pathlib import Path

RUNS_DIR = Path("runs")
REGISTRY_NAME = ".batches.sqlite3"


def scan_runs(root: Path) -> list[Path]:
    """Discover batch directories, oldest first and by name for timestamp ties."""
    found = []
    try:
        for path in root.iterdir():
            if (path / "batch.pkl").is_file():
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
    base = (RUNS_DIR if root is None else Path(root)).resolve()
    candidates = scan_runs(base)
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

    new_names = [(path.name,) for path in candidates if path.name not in known]
    if new_names:
        with closing(sqlite3.connect(registry, timeout=30)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "CREATE TABLE IF NOT EXISTS batches (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE)"
            )
            db.executemany(
                "INSERT OR IGNORE INTO batches (id, name) "
                "SELECT COALESCE(MAX(id), -1) + 1, ? FROM batches",
                new_names,
            )
            known = _registered(db)
    return {known[path.name]: path for path in candidates}
