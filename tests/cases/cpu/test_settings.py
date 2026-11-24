from pathlib import Path

import pytest

import expman
from expman.settings import project_settings


def test_python_overrides_are_validated_before_any_value_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with project_settings():
        assert expman.get("mps_clients_per_gpu") == 0
        expman.set(
            batches_dir="my-results", poll_interval=2, web_port=0, mps_clients_per_gpu=0
        )
        assert expman.get("batches_dir") == Path("my-results")
        assert expman.get("poll_interval") == 2.0
        for invalid in (
            {"poll_interval": float("nan")},
            {"web_port": True},
            {"worker_threads": 0},
            {"retries": -1},
            {"mps_clients_per_gpu": -1},
            {"coverage": 1.0},
            {"live_enabled": "false"},
            {"batches_dir": ""},
            {"stop_fifo": "alternate.fifo"},
        ):
            with pytest.raises(ValueError):
                expman.set(web_host="changed", **invalid)
        assert expman.get("web_host") == "127.0.0.1"
        assert expman.get("web_port") == 0
        assert expman.get("retries") == 1
        assert expman.get("coverage") == 0.8
        with pytest.raises(ValueError, match="unknown setting"):
            expman.get("stop_fifo")


def test_only_working_directory_settings_apply_and_python_overrides_win(
    tmp_path, monkeypatch
):
    project = tmp_path / "experiment"
    project.mkdir()
    monkeypatch.chdir(project)
    (tmp_path / "pyproject.toml").write_text("[tool.expman]\nweb_port = 9850\n")

    with project_settings():
        assert expman.get("web_port") == 8765
        expman.set(web_port=8765)
        (project / "pyproject.toml").write_text(
            "[tool.expman]\nweb_port = 9000\nworker_threads = 2\n"
        )
        with project_settings():
            assert expman.get("web_port") == 8765
            assert expman.get("worker_threads") == 2
            expman.set(worker_threads=3)
        assert expman.get("worker_threads") == 4


def test_invalid_project_identifies_the_file_and_leaves_settings_intact(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "pyproject.toml"
    source.write_text("[tool.expman]\nlive_enabled = false\nweb_port = -1\n")
    with (
        pytest.raises(ValueError, match=r"pyproject\.toml: web_port"),
        project_settings(),
    ):
        pytest.fail("invalid settings must be rejected before running the command")
    assert expman.get("live_enabled") is True
