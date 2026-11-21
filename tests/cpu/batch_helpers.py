"""Build scheduler fixtures without importing a user pipeline module."""

from expman.config.loader import _load_configs
from expman.scheduling.batch import _Batch
from expman.storage.batch import BatchStore


def make_batch(pipeline, config, *, output_dir):
    configs = [config] if isinstance(config, dict) else _load_configs(config)
    return _Batch.create(
        pipeline, configs, output_dir=output_dir, pipeline_spec="pipeline:train"
    )


def restore_batch(pipeline, directory):
    store = BatchStore(directory.resolve())
    return _Batch.resume(pipeline, store, store.read_resumable_manifest())
