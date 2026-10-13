"""Stable integer IDs for the batches living under ``runs/``.

The CLI and the dashboard resolve a batch the same way, so the registry
lives here instead of inside one of them.  The file is advisory: a missing
or damaged ``runs/.batches.json`` is rebuilt by scanning the directory, and
the ID of a batch that disappears stays reserved so the remaining IDs never
shift under a running process.

Every entry point takes an optional root so the dashboard can be pointed at
another directory tree, which is also what the tests do.
"""

import json
from pathlib import Path

RUNS_DIR = Path("runs")
REGISTRY_NAME = ".batches.json"


def _root(root: Path | None) -> Path:
    return RUNS_DIR if root is None else Path(root)


def load_registry(root: Path | None = None) -> dict:
    """Read the ID registry, tolerating a missing or damaged file."""
    try:
        data = json.loads((_root(root) / REGISTRY_NAME).read_text())
    except (OSError, ValueError):
        return {"next_id": 0, "batches": {}}
    if not isinstance(data, dict) or not isinstance(data.get("batches"), dict):
        return {"next_id": 0, "batches": {}}
    if not isinstance(data.get("next_id"), int):
        data["next_id"] = len(data["batches"])
    return data


def save_registry(registry: dict, root: Path | None = None) -> None:
    base = _root(root)
    base.mkdir(parents=True, exist_ok=True)
    (base / REGISTRY_NAME).write_text(json.dumps(registry, indent=2))


def is_batch_dir(path: Path) -> bool:
    return path.is_dir() and (path / "batch.pkl").exists()


def scan_runs(registry: dict, root: Path | None = None) -> bool:
    """Register unregistered batches, oldest first; True when anything changed."""
    try:
        base = _root(root).resolve()
        candidates = sorted(
            (path for path in base.iterdir() if is_batch_dir(path)),
            key=lambda path: path.stat().st_mtime,
        )
    except OSError:
        return False
    known = set(registry["batches"].values())
    changed = False
    for path in candidates:
        if path.name not in known:
            registry["batches"][str(registry["next_id"])] = path.name
            registry["next_id"] += 1
            changed = True
    return changed


def ensure_ids(root: Path | None = None) -> dict[int, Path]:
    """Assign stable IDs to batches; return the ID -> directory map."""
    registry = load_registry(root)
    if scan_runs(registry, root):
        save_registry(registry, root)
    base = _root(root).resolve()
    return {
        int(batch_id): base / name
        for batch_id, name in registry["batches"].items()
        if (base / name).is_dir()
    }
