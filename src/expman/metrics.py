"""Transactional scalar metrics stored independently of execution snapshots."""

import json
import math
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

from .storage import StorageError


def flatten_metrics(metrics: Mapping[str, Any]) -> list[tuple[str, float]]:
    """Validate the entire tree and encode unambiguous segmented metric paths."""
    rows = []
    active: set[int] = set()

    def visit(node: Mapping, path: tuple[str, ...]) -> None:
        if id(node) in active:
            raise ValueError("metrics must not contain cycles")
        active.add(id(node))
        try:
            for key, value in node.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError("metric names must be nonempty strings")
                child = (*path, key)
                if isinstance(value, Mapping):
                    visit(value, child)
                else:
                    if isinstance(value, bool) or not isinstance(value, Real):
                        raise ValueError("metric values must be finite real numbers")
                    try:
                        scalar = float(value)
                    except (OverflowError, ValueError) as error:
                        raise ValueError(
                            "metric values must be finite real numbers"
                        ) from error
                    if not math.isfinite(scalar):
                        raise ValueError("metric values must be finite real numbers")
                    rows.append((json.dumps(child, ensure_ascii=False), scalar))
        finally:
            active.remove(id(node))

    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping")
    visit(metrics, ())
    return rows


class MetricStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def log(
        self,
        run_id: str,
        stage_path: tuple[int, ...],
        step: int | None,
        attempt: int,
        rows: list[tuple[str, float]],
    ) -> None:
        if not rows:
            return
        timestamp = datetime.now(timezone.utc).isoformat()
        # A non-null sentinel makes summary metrics participate in uniqueness.
        parameters = [
            (
                run_id,
                json.dumps(stage_path),
                -1 if step is None else step,
                path,
                value,
                attempt,
                timestamp,
            )
            for path, value in rows
        ]
        try:
            with (
                closing(sqlite3.connect(self.path, timeout=30)) as connection,
                connection,
            ):
                connection.execute(
                    """CREATE TABLE IF NOT EXISTS metrics (
                        run_id TEXT NOT NULL,
                        stage_path TEXT NOT NULL,
                        step INTEGER NOT NULL,
                        metric_path TEXT NOT NULL,
                        value REAL NOT NULL,
                        attempt INTEGER NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (run_id, stage_path, step, metric_path)
                    )"""
                )
                connection.executemany(
                    """INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (run_id, stage_path, step, metric_path)
                    DO UPDATE SET value=excluded.value, attempt=excluded.attempt,
                                  updated_at=excluded.updated_at""",
                    parameters,
                )
        except (sqlite3.Error, OSError, OverflowError) as error:
            raise StorageError(
                f"could not save metrics to {self.path}: {error}"
            ) from error
