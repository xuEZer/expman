"""Files shared by the batch runner and off-process readers."""

from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from expman.estimation.estimate import TimeEstimate, valid_seconds, validate_coverage
from expman.storage.base import (
    PickleSerializer,
    Serializer,
    StorageError,
    read_record,
    write_record,
)

BATCH_FORMAT = 2


class BatchStore:
    def __init__(self, directory: Path, serializer: Serializer | None = None):
        self.directory = directory
        self.serializer = PickleSerializer() if serializer is None else serializer

    def read_manifest(self) -> dict:
        manifest = read_record(self.directory / "batch.pkl", self.serializer)
        if (
            not isinstance(manifest, dict)
            or manifest.get("version") != BATCH_FORMAT
            or not isinstance(manifest.get("revision"), str)
            or not isinstance(manifest.get("experiments"), list)
            or not isinstance(manifest.get("queue"), list)
        ):
            raise StorageError("unsupported or invalid batch manifest")
        return manifest

    def read_resumable_manifest(self) -> dict:
        manifest = self.read_manifest()
        spec = manifest.get("pipeline_spec")
        if not isinstance(spec, str) or not spec:
            raise StorageError(f"{self.directory} has no recorded pipeline location")
        return manifest

    def save_manifest(self, manifest: dict) -> str:
        revision = uuid4().hex
        write_record(
            self.directory / "batch.pkl",
            {**manifest, "version": BATCH_FORMAT, "revision": revision},
            self.serializer,
        )
        return revision

    def load_config(self, run_id: str) -> dict:
        if (
            not isinstance(run_id, str)
            or not run_id
            or run_id in (".", "..")
            or Path(run_id).name != run_id
        ):
            raise StorageError("invalid experiment ID")
        config = read_record(
            self.directory / "experiments" / run_id / "config.pkl", self.serializer
        )
        if not isinstance(config, dict):
            raise StorageError("invalid experiment configuration")
        return config

    def save_timing(self, elapsed: float, estimate: TimeEstimate | None) -> None:
        saved = asdict(estimate) if estimate and estimate.known else None
        write_record(
            self.directory / "timing.pkl",
            {"version": 1, "elapsed_seconds": elapsed, "estimate": saved},
            self.serializer,
        )

    def read_timing(self) -> tuple[float, TimeEstimate | None] | None:
        path = self.directory / "timing.pkl"
        if not path.exists():
            return None
        record = read_record(path, self.serializer)
        if (
            not isinstance(record, dict)
            or record.get("version") != 1
            or not valid_seconds(record.get("elapsed_seconds"))
            or "estimate" not in record
        ):
            raise StorageError("invalid Batch timing record")
        saved = record["estimate"]
        if saved is None:
            return record["elapsed_seconds"], None
        try:
            estimate = TimeEstimate(**saved)
            validate_coverage(estimate.coverage)
            if not estimate.known:
                raise ValueError("invalid estimate bounds")
        except (TypeError, ValueError) as error:
            raise StorageError("invalid saved time estimate") from error
        return record["elapsed_seconds"], estimate
