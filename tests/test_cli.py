"""Command-line interface: pipeline resolution, run, status and stop."""

import signal
import subprocess
import sys
import time

import pytest

import expman
from expman import cli
from expman.cli import _resolve_pipeline, main
from expman.storage import PickleSerializer, read_record

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
        return {"interrupt_once": True, "max_iter": True}

    def init(self, ctx):
        self.total = 0

    def loop(self, data, ctx, index, max_iter=Stage.cfg("max_iter")):
        self.total += data[index]
        if ctx.cfg["interrupt_once"] and ctx.attempt == 1 and index == 1:
            raise KeyboardInterrupt()
        return self.total


train = Pipeline([LoadValues, Accumulate])
stages = [LoadValues, Accumulate]


def factory():
    return Pipeline([LoadValues, Accumulate])


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

SLOW_MODULE_NAME = "cli_pipeline_slow"
SLOW_MODULE_SOURCE = '''\
from time import sleep

from expman import Pipeline, Stage


class Slow(Stage):
    """Runs until break_loop; long max_iter keeps the window open for stop."""

    @classmethod
    def config_dependencies(cls, cfg):
        return {"max_iter": True}

    def loop(self, data, ctx, index, max_iter=Stage.cfg("max_iter")):
        sleep(0.1)
        if index >= 5:
            self.break_loop()
        return index


train = Pipeline([Slow])
'''


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


def write_config(tmp_path, *, interrupt_once=False, max_iter=3):
    path = tmp_path / "experiments.yaml"
    path.write_text(
        "device: [0]\nseed: 0\nvalues: [1, 2, 3]\n"
        f"max_iter: {max_iter}\n"
        f"interrupt_once: {'true' if interrupt_once else 'false'}\n"
    )
    return path


@pytest.mark.parametrize("attr", ["stages", "factory"])
def test_pipeline_resolution_forms_run(tmp_path, pipeline_module, attr):
    config = write_config(tmp_path)
    output = tmp_path / f"batch-{attr}"

    exit_code = main(
        ["run", f"{pipeline_module}:{attr}", str(config), "-o", str(output)]
    )

    assert exit_code == 0
    manifest = read_record(output / "batch.pkl", PickleSerializer())
    assert manifest["pipeline_spec"] == {"module": pipeline_module, "attr": attr}


def test_run_default_output_dir_under_runs(tmp_path, pipeline_module):
    """Without -o the batch lives at runs/<pipeline-attr>_<config-stem>."""
    config = write_config(tmp_path)

    assert main(["run", f"{pipeline_module}:train", str(config)]) == 0
    assert (tmp_path / "runs" / "train_experiments" / "batch.pkl").exists()

    # A second identical run must not collide; it gets a numeric suffix.
    assert main(["run", f"{pipeline_module}:train", str(config)]) == 0
    assert (tmp_path / "runs" / "train_experiments_2" / "batch.pkl").exists()


def test_run_reads_settings_set_by_the_pipeline_module(
    tmp_path, monkeypatch, settings_restored
):
    """expman.set() in the pipeline module replaces the removed CLI flags."""
    (tmp_path / f"{SETTINGS_MODULE_NAME}.py").write_text(SETTINGS_MODULE_SOURCE)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    captured = {}
    real_batch = cli.Batch

    def spy(pipeline, cfg, **kwargs):
        captured.update(kwargs)
        return real_batch(pipeline, cfg, **kwargs)

    original_batch = cli.Batch
    cli.Batch = spy
    try:
        exit_code = main(
            ["run", f"{SETTINGS_MODULE_NAME}:train", str(write_config(tmp_path))]
        )
    finally:
        cli.Batch = original_batch
        sys.modules.pop(SETTINGS_MODULE_NAME, None)

    assert exit_code == 0
    assert captured["max_retries"] == 3
    assert captured["estimate_coverage"] == 0.9


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
    assert manifest["pipeline_spec"] == {
        "module": pipeline_module,
        "attr": "train",
    }
    assert capsys.readouterr().out.count("succeeded") == 1

    # A completed Batch has no pending work; resuming it succeeds immediately.
    assert main(["resume", str(output)]) == 0


def test_stop_terminates_and_resume_completes(tmp_path, capsys, monkeypatch):
    """expman stop signals the running process; it persists and exits 143."""
    (tmp_path / f"{SLOW_MODULE_NAME}.py").write_text(SLOW_MODULE_SOURCE)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    config = tmp_path / "experiments.yaml"
    config.write_text("device: [0]\nseed: 0\nmax_iter: 10000\n")
    output = tmp_path / "batch"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "expman",
            "run",
            f"{SLOW_MODULE_NAME}:train",
            str(config),
            "-o",
            str(output),
        ],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        pid_file = output / "expman.pid"
        for _ in range(200):
            if pid_file.exists():
                break
            if process.poll() is not None:
                raise AssertionError("run exited before the PID record appeared")
            time.sleep(0.05)
        assert pid_file.exists()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=60) == 143
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=30)
    assert "Stopped; resume with: expman resume" in process.stderr.read()
    assert not pid_file.exists()

    assert main(["resume", str(output)]) == 0
    assert capsys.readouterr().out.count("succeeded") == 1


def test_status_reports_progress(tmp_path, pipeline_module, capsys):
    config = write_config(tmp_path)
    output = tmp_path / "batch"
    main(["run", f"{pipeline_module}:train", str(config), "-o", str(output)])
    capsys.readouterr()

    assert main(["status", str(output)]) == 0
    report = capsys.readouterr().out
    assert f"Batch {output}" in report
    assert "succeeded" in report
    assert "elapsed" in report


def test_status_overview_watches_runs_with_ids(
    tmp_path, pipeline_module, capsys, monkeypatch
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


def test_stop_and_resume_accept_batch_ids(tmp_path, pipeline_module, capsys):
    config = write_config(tmp_path)
    main(["run", f"{pipeline_module}:train", str(config)])
    capsys.readouterr()

    # ID 0 is the batch just created; a completed batch resumes immediately.
    assert main(["resume", "0"]) == 0
    assert main(["stop", "0"]) == 1

    with pytest.raises(SystemExit, match="unknown batch ID"):
        main(["stop", "7"])


def test_stop_without_live_pid_reports_and_cleans_up(tmp_path):
    output = tmp_path / "batch"
    output.mkdir()
    assert main(["stop", str(output)]) == 1

    dead = subprocess.Popen(["true"])
    dead.wait()
    (output / "expman.pid").write_text(str(dead.pid))

    assert main(["stop", str(output)]) == 1
    assert not (output / "expman.pid").exists()


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
