from expman._memory_model import LOCAL_NEIGHBORS, LocalQuantileEstimator, PeakCeiling
from expman._time_model import Encoding, encode, variable_distance


def rows(values):
    """Feature rows without variable grouping: the hashed-column fallback."""
    return Encoding([[1.0, value] for value in values], None)


def test_estimate_is_bounded_by_the_finite_local_samples():
    indices = {f"run-{index}": index for index in range(6)}
    estimator = LocalQuantileEstimator(rows(range(6)), indices, default=1024.0)
    for index, peak in enumerate((300.0, 340.0, 375.0, 410.0, 450.0)):
        estimator.record(f"run-{index}", peak)

    estimate = estimator.estimate("run-5")

    assert 300.0 <= estimate <= 450.0


def test_capped_sample_has_one_finite_retry_bump():
    indices = {"retry": 0, "other": 1}
    estimator = LocalQuantileEstimator(rows([0, 0]), indices, default=1024.0)
    estimator.record("retry", 800.0, capped=True)

    assert estimator.estimate("retry") == 1200.0
    assert estimator.estimate("other") == 1200.0


def test_zero_gpu_telemetry_is_a_real_sample_when_requested():
    indices = {"completed": 0, "pending": 1}
    estimator = LocalQuantileEstimator(
        rows([0, 0]), indices, default=1024.0, include_zero=True
    )
    estimator.record("completed", 0.0)

    assert estimator.estimate("pending") == 0.0


def test_the_peak_model_draws_neighbours_in_the_shared_configuration_distance():
    """A peak estimate is only as good as the neighborhood it was drawn from.

    Both Stage models must therefore agree on what "nearby" means, so this pins
    the peak model to the per-variable distance the duration model already uses.
    """
    configs = [
        {"tag": "a" if index % 3 else "b", "steps": index + 1} for index in range(12)
    ]
    encoding = encode(configs)
    assert encoding.variables is not None
    indices = {f"run-{index}": index for index in range(len(configs))}
    peaks = {f"run-{index}": 100.0 * (index + 1) for index in range(len(configs))}
    estimator = LocalQuantileEstimator(encoding, indices, default=0.0)
    for run_id, peak in peaks.items():
        estimator.record(run_id, peak)

    query = indices["run-11"]
    nearest = sorted(
        peak
        for _run_id, peak in sorted(
            peaks.items(),
            key=lambda item: variable_distance(
                encoding, encoding.rows[query], encoding.rows[indices[item[0]]]
            ),
        )[:LOCAL_NEIGHBORS]
    )

    assert sorted(estimator._nearby(query)) == nearest


def test_rebuild_preserves_capped_history_bump():
    ceiling = PeakCeiling.from_history(
        [
            {"peak_kb": 500.0, "capped": False},
            {"peak_kb": 800.0, "capped": True},
            {"peak_kb": float("inf"), "capped": False},
        ]
    )

    assert ceiling.ceiling_kb == 1200.0
