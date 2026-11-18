import pytest

from expman import Batch, Pipeline, Status, StorageError


def test_empty_pipeline_finishes_and_a_stale_session_cannot_overwrite_it(tmp_path):
    directory = tmp_path / "batch"
    pipeline = Pipeline([])
    original = Batch(pipeline, {"device": [0], "seed": 42}, output_dir=directory)
    stale = Batch.resume(pipeline, directory)

    (result,) = original.run(progress=False)
    assert result.status is Status.SUCCEEDED
    assert result.output is None and len(result.attempts) == 1
    with pytest.raises(StorageError, match="batch records changed"):
        stale.run(progress=False)

    reopened = Batch.resume(pipeline, directory)
    assert reopened.results == (result,)
    estimate = reopened.estimate()
    assert (estimate.lower_seconds, estimate.upper_seconds) == (0.0, 0.0)
