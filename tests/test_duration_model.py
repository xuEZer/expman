import pytest

from expman._duration_model import DurationModel, StageDuration, variable_distance
from expman._time_model import encode


def build(configs, durations, *, coverage=0.8, censored=()):
    encoding = encode(configs)
    indices = {f"run-{index}": index for index in range(len(configs))}
    model = DurationModel(encoding, indices, coverage=coverage)
    for index, seconds in enumerate(durations):
        model.record(f"run-{index}", seconds, censored=index in censored)
    return model, indices


def test_a_stage_measured_once_reports_exactly_what_it_measured():
    model, _indices = build([{"steps": 1}], [12.5])

    assert model.predict("run-0") == StageDuration(12.5, 12.5, 12.5)


def test_a_stage_without_a_completed_sample_has_no_estimate():
    model, _indices = build([{"steps": 1}], [12.5], censored={0})

    assert model.predict("run-0") is None


def test_unknown_run_id_has_no_estimate():
    model, _indices = build([{"steps": 1}], [12.5])

    assert model.predict("absent") is None


def test_doubling_every_duration_doubles_the_estimate():
    """Duration responds multiplicatively, so the model has to as well."""
    configs = [{"steps": value} for value in (1, 2, 3, 4, 5)]
    durations = (10.0, 20.0, 30.0, 40.0, 50.0)
    plain, indices = build(configs, durations)
    doubled, _other = build(configs, [value * 2 for value in durations])

    for run_id in indices:
        assert doubled.predict(run_id).center == pytest.approx(
            plain.predict(run_id).center * 2
        )


def test_a_distant_sample_still_shrinks_the_estimate():
    """No cutoff: the far sample keeps weight instead of being dropped."""
    model, _indices = build([{"steps": 1}, {"steps": 1000}], [10.0, 1000.0])

    predicted = model.predict("run-0")

    assert 10.0 < predicted.center < 1000.0
    # The nearer configuration leads, but not by discarding the other one.
    assert predicted.center == pytest.approx(56.9, rel=0.05)


def test_a_multi_level_category_weighs_once_not_once_per_level():
    """A two-column one-hot group must not outweigh a numeric value."""
    configs = [
        {"model": "a", "steps": 1},
        {"model": "b", "steps": 1},
        {"model": "a", "steps": 2},
    ]
    encoding = encode(configs)

    assert [(item.kind, len(item.columns)) for item in encoding.variables] == [
        ("categorical", 2),
        ("numeric", 1),
    ]
    rows = encoding.rows
    changed_category = variable_distance(encoding, rows[0], rows[1])
    changed_number = variable_distance(encoding, rows[0], rows[2])
    assert changed_category == pytest.approx(0.5)
    assert changed_number == pytest.approx(changed_category)


def test_a_wide_configuration_drops_variable_grouping():
    configs = [{f"k{index}": value for index in range(60)} for value in (1, 2)]
    encoding = encode(configs)

    assert encoding.variables is None
    assert len(encoding.rows[0]) == 49
    assert variable_distance(encoding, encoding.rows[0], encoding.rows[0]) == 0.0
    assert variable_distance(encoding, encoding.rows[0], encoding.rows[1]) > 0.0


def test_the_interval_brackets_the_center_and_widens_with_coverage():
    configs = [{"steps": value} for value in (1, 2, 3, 4, 5, 6)]
    durations = (10.0, 12.0, 11.0, 18.0, 15.0, 13.0)
    narrow, _first = build(configs, durations, coverage=0.5)
    wide, _second = build(configs, durations, coverage=0.95)

    for model in (narrow, wide):
        estimate = model.predict("run-0")
        assert estimate.lower <= estimate.center <= estimate.upper
    assert (wide.predict("run-0").upper - wide.predict("run-0").lower) > (
        narrow.predict("run-0").upper - narrow.predict("run-0").lower
    )


def test_a_censored_sample_cannot_inflate_the_upper_side():
    """A cancelled attempt measured a lower bound, so it stays out of the top."""
    configs = [{"steps": 1}] * 5
    measured = [10.0, 11.0, 12.0, 13.0, 1000.0]
    censored, _first = build(configs, measured, censored={4})
    ignored, _second = build(configs, measured)

    upper = censored.predict("run-0").upper
    assert upper < 10 * max(measured[:4])
    assert upper < ignored.predict("run-0").upper / 10


def test_a_censored_sample_widens_the_lower_side():
    """An attempt cancelled early is a lower bound, so it loosens the floor."""
    configs = [{"steps": 1}] * 5
    measured = [10.0, 11.0, 12.0, 13.0, 1.0]
    censored, _first = build(configs, measured, censored={4})
    completed, _second = build(configs, measured[:4])

    assert censored.predict("run-0").lower < completed.predict("run-0").lower


def test_samples_outside_the_positive_finite_range_are_rejected():
    model, _indices = build([{"steps": 1}], [12.5])
    model.record("run-0", 0.0)
    model.record("run-0", -1.0)
    model.record("run-0", float("inf"))
    model.record("run-0", True)

    assert model.predict("run-0") == StageDuration(12.5, 12.5, 12.5)
