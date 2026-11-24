"""Create, restore and wait for batches in scheduler and CLI tests."""

from time import monotonic, sleep

import pytest

from expman.config.loader import _load_configs
from expman.observation.status import RecordedBatch
from expman.scheduling.batch import _Batch
from expman.scheduling.control import request_stop
from expman.storage.base import StorageError, run_lock
from expman.storage.batch import BatchStore


def make_batch(pipeline, config, *, output_dir):
    configs = [config] if isinstance(config, dict) else _load_configs(config)
    return _Batch.create(
        pipeline, configs, output_dir=output_dir, pipeline_spec="pipeline:train"
    )


def restore_batch(pipeline, directory):
    store = BatchStore(directory.resolve())
    return _Batch.resume(pipeline, store, store.read_resumable_manifest())


def wait_for_batch(directory, *, timeout=30):
    """Read the final state after the background scheduler releases its lock."""
    deadline = monotonic() + timeout
    while True:
        try:
            with run_lock(directory):
                break
        except StorageError:
            if monotonic() >= deadline:
                request_stop(directory)
                pytest.fail(f"batch still owns its execution lock: {directory}")
            sleep(0.1)
    return RecordedBatch.read(directory)
