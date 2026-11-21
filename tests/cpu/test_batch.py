import pytest
from batch_helpers import make_batch, restore_batch

from expman import Pipeline, Status, StorageError


def test_empty_pipeline_finishes_and_a_stale_session_cannot_overwrite_it(tmp_path):
    directory = tmp_path / "batch"
    pipeline = Pipeline([])
    original = make_batch(pipeline, {"device": [0], "seed": 42}, output_dir=directory)
    stale = restore_batch(pipeline, directory)

    (result,) = original.run()
    assert result.status is Status.SUCCEEDED
    assert len(result.attempts) == 1
    with pytest.raises(StorageError, match="batch records changed"):
        stale.run()

    reopened = restore_batch(pipeline, directory)
    assert reopened.results == (result,)
    estimate = reopened.estimate()
    assert (estimate.lower_seconds, estimate.upper_seconds) == (0.0, 0.0)
