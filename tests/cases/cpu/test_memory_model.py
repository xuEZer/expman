from expman.estimation.memory_model import (
    LocalQuantileEstimator,
    PeakCeiling,
)
from expman.estimation.time_model import Encoding, encode
from expman.settings import settings


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


def test_estimates_follow_nearby_configurations(monkeypatch):
    monkeypatch.setattr(settings, "local_neighbors", 2)
    monkeypatch.setattr(settings, "local_min_samples", 4)
    models = ("large", "small", "large", "small", "large", "small")
    encoding = encode([{"model": model} for model in models])
    indices = {f"run-{index}": index for index in range(len(models))}
    estimator = LocalQuantileEstimator(encoding, indices, default=1024)
    for index, peak in enumerate((1000, 10, 1000, 10)):
        estimator.record(f"run-{index}", peak)

    assert estimator.estimate("run-4") == 1000
    assert estimator.estimate("run-5") == 10


def test_rebuild_preserves_capped_history_bump():
    ceiling = PeakCeiling.from_history(
        [
            {"peak_kb": 500.0, "capped": False},
            {"peak_kb": 800.0, "capped": True},
            {"peak_kb": float("inf"), "capped": False},
        ]
    )

    assert ceiling.ceiling_kb == 1200.0
