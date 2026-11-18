"""Command-line interface: pipeline resolution, run, status and stop."""

import os
import signal
import subprocess
import sys

import pytest

import expman
from expman import cli
from expman.cli import _resolve_pipeline, main
from expman.process_control import STOP_FIFO, is_running, stop_on_sigterm
from expman.storage.base import PickleSerializer, read_record, run_lock

MODULE_NAME = "cli_pipeline_under_test"

MODULE_SOURCE = """\
from expman import Pipeline, Stage


class LoadValues(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {"values": True}

    def process(self, data, ctx):
        return list(ctx.cfg["values"])


class Accumulate(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {"interrupt_once": True}

    def process(self, data, ctx):
        total = ctx.checkpoint_state
        if total is None:
            total = 0
        for index in ctx.resume_range(0, len(data)):
            total += data[index]
            ctx.checkpoint(index + 1, total)
            if ctx.cfg["interrupt_once"] and ctx.attempt == 1 and index == 1:
                raise KeyboardInterrupt()
        return total


train = Pipeline([LoadValues, Accumulate])
empty = Pipeline([])
not_a_pipeline = 42
"""

SETTINGS_MODULE_NAME = "cli_pipeline_settings"

SETTINGS_MODULE_SOURCE = """\
import expman

expman.set(retries=3, coverage=0.9)

from expman import Pipeline, Stage


class Trivial(Stage):
    def process(self, data, ctx):
        return data


train = Pipeline([Trivial])
"""


@pytest.fixture()
def pipeline_module(tmp_path, monkeypatch):
    """Publish the pipeline module in an isolated working directory."""
    (tmp_path / f"{MODULE_NAME}.py").write_text(MODULE_SOURCE)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    try:
        yield MODULE_NAME
    finally:
        sys.modules.pop(MODULE_NAME, None)


@pytest.fixture()
def settings_restored():
    """Keep the process-wide expman settings from leaking between tests."""
    try:
        yield
    finally:
        expman.set(retries=1, coverage=0.8)


@pytest.fixture
def prepare_batch_only(monkeypatch):
    """Let CLI commands persist a real batch without starting its worker processes."""
    monkeypatch.setattr(cli, "_execute", lambda batch, **options: 0)


def write_config(tmp_path, *, interrupt_once=False):
    path = tmp_path / "experiments.yaml"
    path.write_text(
        "device: [0]\nseed: 0\nvalues: [1, 2, 3]\n"
        f"interrupt_once: {'true' if interrupt_once else 'false'}\n"
    )
    return path


def test_run_default_output_dir_under_runs(
    tmp_path, pipeline_module, prepare_batch_only
):
    """Without -o the batch lives at runs/<pipeline-attr>_<config-stem>."""
    config = write_config(tmp_path)

    assert main(["run", f"{pipeline_module}:train", str(config)]) == 0
    assert (tmp_path / "runs" / "train_experiments" / "batch.pkl").exists()

    # A second identical run must not collide; it gets a numeric suffix.
    assert main(["run", f"{pipeline_module}:train", str(config)]) == 0
    assert (tmp_path / "runs" / "train_experiments_2" / "batch.pkl").exists()


def test_run_reads_settings_set_by_the_pipeline_module(
    tmp_path, monkeypatch, settings_restored, prepare_batch_only
):
    (tmp_path / f"{SETTINGS_MODULE_NAME}.py").write_text(SETTINGS_MODULE_SOURCE)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    output = tmp_path / "batch"
    try:
        assert (
            main(
                [
                    "run",
                    f"{SETTINGS_MODULE_NAME}:train",
                    str(write_config(tmp_path)),
                    "-o",
                    str(output),
                ]
            )
            == 0
        )
    finally:
        sys.modules.pop(SETTINGS_MODULE_NAME, None)
    manifest = read_record(output / "batch.pkl", PickleSerializer())
    assert manifest["max_retries"] == 3
    assert manifest["estimate_coverage"] == 0.9


@pytest.mark.simulated_gpu
def test_run_records_pipeline_spec_and_resume_repeats_it(
    tmp_path, pipeline_module, capsys
):
    config = write_config(tmp_path)
    output = tmp_path / "batch"

    exit_code = main(
        ["run", f"{pipeline_module}:train", str(config), "-o", str(output)]
    )

    assert exit_code == 0
    manifest = read_record(output / "batch.pkl", PickleSerializer())
    assert manifest["pipeline_spec"] == f"{pipeline_module}:train"
    assert capsys.readouterr().out.count("succeeded") == 1

    assert main(["resume", str(output)]) == 0
    restored = expman.Batch.resume(
        _resolve_pipeline(f"{pipeline_module}:train"), output
    )
    assert restored.results[0].output == 6


@pytest.mark.simulated_gpu
def test_run_and_resume_report_terminal_failure_to_the_shell(
    tmp_path, pipeline_module, capsys
):
    config = tmp_path / "failing.yaml"
    config.write_text("device: [0]\nseed: 0\nvalues: [1, bad]\ninterrupt_once: false\n")
    output = tmp_path / "batch"
    assert (
        main(["run", f"{pipeline_module}:train", str(config), "-o", str(output)]) == 1
    )
    assert ": failed" in capsys.readouterr().out
    assert main(["resume", str(output)]) == 1
    assert ": failed" in capsys.readouterr().out


def test_competing_cli_resume_preserves_the_owners_stop_channel(
    tmp_path, monkeypatch, pipeline_module
):
    output = tmp_path / "batch"
    endpoint = output / STOP_FIFO
    previous_handler = signal.getsignal(signal.SIGTERM)

    def attempt_resume(batch):
        inode = endpoint.stat().st_ino
        assert is_running(output)
        challenger = subprocess.run(
            [sys.executable, "-m", "expman", "resume", str(output)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert challenger.returncode != 0
        assert "already running" in challenger.stderr
        assert endpoint.stat().st_ino == inode and is_running(output)

    monkeypatch.setattr(expman.Batch, "_start_live", attempt_resume)
    assert (
        main(
            [
                "run",
                f"{pipeline_module}:empty",
                str(write_config(tmp_path)),
                "-o",
                str(output),
            ]
        )
        == 0
    )
    assert not endpoint.exists()
    assert signal.getsignal(signal.SIGTERM) == previous_handler


def test_status_reports_a_queued_run(
    tmp_path, pipeline_module, capsys, prepare_batch_only
):
    config = write_config(tmp_path)
    output = tmp_path / "batch"
    main(["run", f"{pipeline_module}:train", str(config), "-o", str(output)])
    capsys.readouterr()

    assert main(["status", str(output)]) == 0
    report = capsys.readouterr().out
    assert f"Batch {output}" in report
    assert "pending" in report
    assert "stage 1/2" in report
    assert "elapsed" in report


def test_status_overview_watches_runs_with_ids(
    tmp_path, pipeline_module, capsys, monkeypatch, prepare_batch_only
):
    """Bare expman status lists every runs/ batch with a stable integer ID."""
    config = write_config(tmp_path)
    main(["run", f"{pipeline_module}:train", str(config)])
    main(["run", f"{pipeline_module}:train", str(config)])

    calls = {"count": 0}

    def interrupt_on_second_frame(_seconds):
        calls["count"] += 1
        if calls["count"] > 1:
            raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "sleep", interrupt_on_second_frame)
    assert main(["status"]) == 0
    report = capsys.readouterr().out
    assert " 0  runs/train_experiments" in report
    assert " 1  runs/train_experiments_2" in report


def test_stop_and_resume_accept_batch_ids(
    tmp_path, pipeline_module, capsys, prepare_batch_only
):
    config = write_config(tmp_path)
    main(["run", f"{pipeline_module}:train", str(config)])
    capsys.readouterr()

    # The registered ID resolves to the same batch for both commands.
    assert main(["resume", "0"]) == 0
    assert main(["stop", "0"]) == 1

    with pytest.raises(SystemExit, match="unknown batch ID"):
        main(["stop", "7"])


def test_stop_without_a_reader_reports_and_cleans_up(tmp_path):
    output = tmp_path / "batch"
    output.mkdir()
    assert main(["stop", str(output)]) == 1

    endpoint = output / STOP_FIFO
    os.mkfifo(endpoint)

    assert main(["stop", str(output)]) == 1
    assert not endpoint.exists()


@pytest.mark.parametrize("symlink", [False, True])
def test_stop_control_preserves_unrelated_files(tmp_path, symlink):
    endpoint = tmp_path / STOP_FIFO
    contents = tmp_path / "ordinary" if symlink else endpoint
    contents.write_text("untouched")
    if symlink:
        endpoint.symlink_to(contents)
    assert not is_running(tmp_path)
    assert main(["stop", str(tmp_path)]) == 1
    with run_lock(tmp_path), pytest.raises(FileExistsError), stop_on_sigterm(tmp_path):
        pytest.fail("cannot publish over a foreign path")
    assert contents.read_text() == "untouched" and endpoint.exists()


def test_an_old_stop_writer_cannot_interrupt_the_next_execution(tmp_path):
    with run_lock(tmp_path), stop_on_sigterm(tmp_path):
        fd = os.open(tmp_path / STOP_FIFO, os.O_WRONLY | os.O_NONBLOCK)
    try:
        with run_lock(tmp_path), stop_on_sigterm(tmp_path):
            with pytest.raises(BrokenPipeError):
                os.write(fd, b"x")
            assert is_running(tmp_path)
    finally:
        os.close(fd)


def test_resolution_rejects_malformed_and_mistyped_specs(pipeline_module):
    with pytest.raises(SystemExit, match="module:attr"):
        _resolve_pipeline("no-colon-here")
    with pytest.raises(SystemExit, match="could not import"):
        _resolve_pipeline("no_such_pipeline_module_xyz:train")
    with pytest.raises(SystemExit, match="defines no attribute"):
        _resolve_pipeline(f"{pipeline_module}:missing_attr")
    with pytest.raises(SystemExit, match="must be a Pipeline"):
        _resolve_pipeline(f"{pipeline_module}:not_a_pipeline")


def test_settings_validate_values():
    with pytest.raises(ValueError, match="unknown setting"):
        expman.set(bogus=1)
    with pytest.raises(ValueError, match="nonnegative"):
        expman.set(retries=-1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        expman.set(coverage=1.5)
    with pytest.raises(ValueError, match="unknown setting"):
        expman.get("bogus")
    # Failed assignments leave the defaults intact.
    assert expman.get("retries") == 1
    assert expman.get("coverage") == 0.8


def test_resume_rejects_directory_without_pipeline_spec(tmp_path):
    from expman import Batch, Pipeline, Stage

    class Trivial(Stage):
        def process(self, data, ctx):
            return data

    config = tmp_path / "cfg.yaml"
    config.write_text("device: [0]\nseed: 0\n")
    output = tmp_path / "batch"
    Batch(Pipeline([Trivial]), config, output_dir=output)

    with pytest.raises(SystemExit, match="records no pipeline location"):
        main(["resume", str(output)])
