"""Framework-set worker environment: the thread ceiling and the HF mirror."""

import pytest

from expman import environment


@pytest.fixture(autouse=True)
def clean_shell(monkeypatch):
    """Start from a shell that exports none of the keys the policy touches."""
    for name in (
        *environment.THREAD_VARIABLES,
        "HF_ENDPOINT",
        environment.THREADS_OVERRIDE,
        environment.MIRROR_OVERRIDE,
    ):
        monkeypatch.delenv(name, raising=False)


def test_overlay_sets_the_thread_ceiling_and_the_mirror():
    values = environment.overlay()

    assert set(values) == {*environment.THREAD_VARIABLES, "HF_ENDPOINT"}
    assert {values[name] for name in environment.THREAD_VARIABLES} == {
        str(environment.WORKER_THREADS)
    }
    assert values["HF_ENDPOINT"] == environment.HF_MIRROR


def test_an_exported_value_wins(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://example.invalid")
    # Only the exported key is left alone; the rest of the policy still applies.
    assert set(environment.overlay()) == set(environment.THREAD_VARIABLES)

    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    # One exported pool size switches the whole group off: sizing the rest at the
    # framework default would leave the libraries disagreeing with each other.
    assert environment.overlay() == {}


def test_overrides_replace_the_defaults(monkeypatch):
    monkeypatch.setenv(environment.THREADS_OVERRIDE, "2")
    assert environment.overlay()["OMP_NUM_THREADS"] == "2"

    monkeypatch.setenv(environment.THREADS_OVERRIDE, "off")
    assert "OMP_NUM_THREADS" not in environment.overlay()

    monkeypatch.delenv(environment.THREADS_OVERRIDE)
    monkeypatch.setenv(environment.MIRROR_OVERRIDE, "off")
    assert "HF_ENDPOINT" not in environment.overlay()

    monkeypatch.setenv(environment.MIRROR_OVERRIDE, "cn")
    assert environment.overlay()["HF_ENDPOINT"] == environment.HF_MIRROR

    monkeypatch.setenv(environment.MIRROR_OVERRIDE, "hf-mirror.com")
    assert environment.overlay()["HF_ENDPOINT"] == "https://hf-mirror.com"

    monkeypatch.setenv(environment.MIRROR_OVERRIDE, "https://mirror.internal")
    assert environment.overlay()["HF_ENDPOINT"] == "https://mirror.internal"


@pytest.mark.parametrize("value", ["zero", "0", "-1", "1.5"])
def test_invalid_thread_overrides_are_rejected(monkeypatch, value):
    monkeypatch.setenv(environment.THREADS_OVERRIDE, value)
    with pytest.raises(ValueError, match="EXPMAN_THREADS"):
        environment.thread_budget()


@pytest.mark.parametrize("value", ["", "  "])
def test_blank_overrides_are_not_overrides(monkeypatch, value):
    monkeypatch.setenv(environment.THREADS_OVERRIDE, value)
    monkeypatch.setenv(environment.MIRROR_OVERRIDE, value)

    assert environment.thread_budget() == environment.WORKER_THREADS
    assert environment.hf_endpoint() == environment.HF_MIRROR


def test_a_bad_override_fails_before_the_batch_directory_exists(tmp_path, monkeypatch):
    """The mistake is reported while nothing has been created yet."""
    from expman import Batch, Pipeline, Stage

    class Trivial(Stage):
        def process(self, data, ctx):
            return data

    config = tmp_path / "experiments.yaml"
    config.write_text("device: [0]\nseed: 0\n")
    output = tmp_path / "batch"
    monkeypatch.setenv(environment.THREADS_OVERRIDE, "abc")

    with pytest.raises(ValueError, match="EXPMAN_THREADS"):
        Batch(Pipeline([Trivial]), config, output_dir=output)

    assert not output.exists()
