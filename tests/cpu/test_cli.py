"""Command-line pipeline resolution, batch management and stop control."""

import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime

import pytest

import expman
from expman import cli
from expman.cli import main
from expman.observation.registry import ensure_ids
from expman.process_control import STOP_FIFO, is_running, request_stop, stop_on_sigterm
from expman.scheduling.batch import _Batch
from expman.storage.base import run_lock
from expman.storage.batch import BatchStore

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

expman.set(retries=3, coverage=0.9, batches_dir="pipeline-batches")

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


@pytest.fixture
def prepare_batch_only(monkeypatch):
    """Let CLI commands persist a real batch without starting its worker processes."""
    prepared = []

    def capture(batch, **options):
        prepared.append(batch)
        return 0

    monkeypatch.setattr(cli, "_execute", capture)
    return prepared


def write_config(tmp_path, *, interrupt_once=False):
    path = tmp_path / "experiments.yaml"
    path.write_text(
        "device: [0]\nseed: 0\nvalues: [1, 2, 3]\n"
        f"interrupt_once: {'true' if interrupt_once else 'false'}\n"
    )
    return path


@pytest.fixture
def create_cli_batch(tmp_path, pipeline_module, prepare_batch_only):
    def create(name, root):
        config = write_config(tmp_path).rename(tmp_path / f"{name}.yaml")
        assert (
            main(
                [
                    "run",
                    f"{pipeline_module}:empty",
                    str(config),
                    "--batches-dir",
                    str(root),
                ]
            )
            == 0
        )
        return prepare_batch_only[-1].output_dir

    return create


def test_batch_is_not_a_public_python_entry_point():
    assert not hasattr(expman, "Batch")
    assert "Batch" not in expman.__all__


@pytest.mark.parametrize(
    ("options", "parent"), [([], "batches"), (["--batches-dir", "results"], "results")]
)
def test_run_creates_a_named_batch_under_the_output_parent(
    tmp_path, pipeline_module, prepare_batch_only, options, parent, monkeypatch
):
    monkeypatch.setattr(cli, "strftime", datetime(2026, 1, 2, 3, 4, 5).strftime)
    config = write_config(tmp_path).rename(tmp_path / "experiment.large.yaml")
    command = ["run", f"{pipeline_module}:train", str(config), *options]

    assert main(command) == 0
    directory = tmp_path / parent / "train_experiment.large_20260102030405"
    manifest = directory / "batch.pkl"
    original = manifest.read_bytes()
    with pytest.raises(FileExistsError) as collision:
        main(command)
    assert collision.value.filename == str(directory)
    assert manifest.read_bytes() == original
    assert ensure_ids(tmp_path / parent) == {0: directory}


@pytest.mark.parametrize(
    ("options", "parent"),
    [([], "pipeline-batches"), (["--batches-dir", "explicit"], "explicit")],
)
def test_run_reads_settings_set_by_the_pipeline_module(
    tmp_path, monkeypatch, prepare_batch_only, options, parent
):
    (tmp_path / f"{SETTINGS_MODULE_NAME}.py").write_text(SETTINGS_MODULE_SOURCE)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    try:
        assert (
            main(
                [
                    "run",
                    f"{SETTINGS_MODULE_NAME}:train",
                    str(write_config(tmp_path)),
                    *options,
                ]
            )
            == 0
        )
    finally:
        sys.modules.pop(SETTINGS_MODULE_NAME, None)
    manifest = BatchStore(ensure_ids(tmp_path / parent)[0]).read_manifest()
    assert manifest["max_retries"] == 3
    assert manifest["estimate_coverage"] == 0.9


def test_resume_of_a_short_pipeline_location_uses_its_original_module(tmp_path):
    (tmp_path / "pipeline.py").write_text(
        "from expman import Pipeline\ntrain = Pipeline([])\n"
    )
    cli = [sys.executable, "-m", "expman"]
    subprocess.run(
        [*cli, "run", "train", str(write_config(tmp_path))], cwd=tmp_path, check=True
    )
    store = BatchStore(ensure_ids(tmp_path / "batches")[0])
    manifest = store.read_manifest()
    assert manifest["pipeline_spec"] == "pipeline:train"
    manifest["pipeline_spec"] = "train"
    store.save_manifest(manifest)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.expman]\npipeline_module = "different"\n'
    )
    subprocess.run([*cli, "resume", "0"], cwd=tmp_path, check=True)


def test_project_defaults_apply_across_cli_processes_and_explicit_arguments_win(
    tmp_path,
):
    config = write_config(tmp_path)
    project = tmp_path / "pyproject.toml"
    project.write_text(
        '[tool.expman]\npipeline_module = "workflows"\nbatches_dir = "project-batches"\n'
        "web_port = 9301\n"
    )
    (tmp_path / "workflows.py").write_text(
        "from expman import Pipeline\ntrain = Pipeline([])\n"
    )
    cli = [sys.executable, "-m", "expman"]
    for options, parent in [
        ([], "project-batches"),
        (["--batches-dir", "custom"], "custom"),
    ]:
        subprocess.run(
            [*cli, "run", "train", str(config), *options], cwd=tmp_path, check=True
        )
        store = BatchStore(ensure_ids(tmp_path / parent)[0])
        assert store.read_manifest()["pipeline_spec"] == "workflows:train"
    help_text = subprocess.check_output(
        [*cli, "web", "--help"], cwd=tmp_path, text=True
    )
    assert "9301" in help_text

    project.write_text(project.read_text().replace('"workflows"', '"different"'))
    subprocess.run([*cli, "resume", "0"], cwd=tmp_path, check=True)
    subprocess.run([*cli, "delete", "0"], cwd=tmp_path, check=True)
    assert ensure_ids(tmp_path / "project-batches") == {}
    assert 0 in ensure_ids(tmp_path / "custom")


@pytest.mark.simulated_gpu
def test_run_records_pipeline_spec_and_resume_repeats_it(
    tmp_path, pipeline_module, capsys
):
    config = write_config(tmp_path)

    exit_code = main(
        ["run", f"{pipeline_module}:train", str(config), "--batches-dir", str(tmp_path)]
    )

    assert exit_code == 0
    output = ensure_ids(tmp_path)[0]
    store = BatchStore(output)
    manifest = store.read_manifest()
    assert manifest["pipeline_spec"] == f"{pipeline_module}:train"
    stdout = capsys.readouterr().out
    assert "Batch ID: 0" in stdout
    assert f"Batch directory: {output}" in stdout
    assert stdout.count("succeeded") == 1

    assert main(["resume", "0", "--batches-dir", str(tmp_path)]) == 0
    assert ": succeeded" in capsys.readouterr().out
    assert store.read_manifest()["experiments"] == manifest["experiments"]


@pytest.mark.simulated_gpu
def test_run_and_resume_report_terminal_failure_to_the_shell(
    tmp_path, pipeline_module, capsys
):
    config = tmp_path / "failing.yaml"
    config.write_text("device: [0]\nseed: 0\nvalues: [1, bad]\ninterrupt_once: false\n")
    assert (
        main(
            [
                "run",
                f"{pipeline_module}:train",
                str(config),
                "--batches-dir",
                str(tmp_path),
            ]
        )
        == 1
    )
    assert ": failed" in capsys.readouterr().out
    assert main(["resume", "0", "--batches-dir", str(tmp_path)]) == 1
    assert ": failed" in capsys.readouterr().out


def test_competing_cli_resume_preserves_the_owners_stop_channel(
    tmp_path, monkeypatch, pipeline_module
):
    previous_handler = signal.getsignal(signal.SIGTERM)

    def attempt_resume(batch):
        output = batch.output_dir
        endpoint = output / STOP_FIFO
        inode = endpoint.stat().st_ino
        assert is_running(output)
        challenger = subprocess.run(
            [
                sys.executable,
                "-m",
                "expman",
                "resume",
                "0",
                "--batches-dir",
                str(tmp_path),
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert challenger.returncode != 0
        assert "already running" in challenger.stderr
        assert endpoint.stat().st_ino == inode and is_running(output)

    monkeypatch.setattr(_Batch, "_start_live", attempt_resume)
    assert (
        main(
            [
                "run",
                f"{pipeline_module}:empty",
                str(write_config(tmp_path)),
                "--batches-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert not (ensure_ids(tmp_path)[0] / STOP_FIFO).exists()
    assert signal.getsignal(signal.SIGTERM) == previous_handler


@pytest.fixture
def watch_status(monkeypatch, capsys):
    def watch(*args, after_first_frame=None):
        frames = []

        def on_refresh(seconds):
            assert seconds == 2
            frames.append(capsys.readouterr().out)
            if len(frames) == 2:
                raise KeyboardInterrupt
            if after_first_frame is not None:
                after_first_frame()

        monkeypatch.setattr(cli, "sleep", on_refresh)
        capsys.readouterr()
        assert main(["status", *args]) == 0
        assert len(frames) == 2
        return frames

    return watch


def test_status_detail_rereads_the_batch(
    tmp_path, pipeline_module, prepare_batch_only, watch_status
):
    assert main(["run", f"{pipeline_module}:train", str(write_config(tmp_path))]) == 0
    (batch,) = prepare_batch_only
    run_id = batch.experiments[0].run_id

    def start_run():
        batch.state.start(run_id, device=0, stage_index=0)
        batch._save()

    first, second = watch_status("0", after_first_frame=start_run)
    assert f"{run_id}  pending" in first
    assert f"{run_id}  running" in second
    for frame in (first, second):
        assert str(batch.output_dir) in frame
        assert "stage 1/2" in frame
        assert "elapsed" in frame


@pytest.mark.parametrize(
    ("options", "parent"), [([], "batches"), (["--batches-dir", "results"], "results")]
)
def test_status_overview_lists_batches_with_ids(
    tmp_path, pipeline_module, prepare_batch_only, watch_status, options, parent
):
    config = str(write_config(tmp_path))
    main(["run", f"{pipeline_module}:train", config, *options])
    first, second = watch_status(
        *options,
        after_first_frame=lambda: main(
            ["run", f"{pipeline_module}:empty", config, *options]
        ),
    )
    ids = ensure_ids(tmp_path / parent)
    assert f" 0  {parent}/{ids[0].name} " in first
    assert ids[1].name not in first
    assert f" 1  {parent}/{ids[1].name} " in second


def test_id_zero_selects_different_batches_in_different_parents(
    tmp_path, pipeline_module, prepare_batch_only, watch_status
):
    config = str(write_config(tmp_path))
    main(["run", f"{pipeline_module}:train", config])
    main(["run", f"{pipeline_module}:empty", config, "--batches-dir", "other"])
    assert str(ensure_ids(tmp_path / "batches")[0]) in watch_status("0")[0]
    assert (
        str(ensure_ids(tmp_path / "other")[0])
        in watch_status("0", "--batches-dir", "other")[0]
    )


@pytest.mark.parametrize("command", ["status", "resume", "stop", "delete"])
def test_batch_ids_reject_directory_paths(command):
    with pytest.raises(SystemExit) as error:
        main([command, "batches/train_experiments"])
    assert error.value.code == 2


def test_stop_and_resume_accept_batch_ids(
    tmp_path, pipeline_module, capsys, prepare_batch_only
):
    config = write_config(tmp_path)
    main(["run", f"{pipeline_module}:train", str(config)])
    capsys.readouterr()

    # The registered ID resolves to the same batch for both commands.
    assert main(["resume", "0"]) == 0
    assert main(["stop", "0"]) == 1

    for token in ("-1", "7"):
        with pytest.raises(SystemExit, match=f"unknown batch ID {token}"):
            main(["stop", token])


def test_delete_removes_the_batch_and_reuses_the_smallest_free_id(
    tmp_path, create_cli_batch
):
    root = tmp_path / "batches"
    first = create_cli_batch("first", root)
    middle = create_cli_batch("middle", root)
    last = create_cli_batch("last", root)
    assert main(["delete", "1"]) == 0
    assert not middle.exists()
    assert ensure_ids(root) == {0: first, 2: last}

    replacement = create_cli_batch("replacement", root)
    assert ensure_ids(root) == {0: first, 1: replacement, 2: last}


def test_delete_releases_a_record_left_by_manual_removal(tmp_path, create_cli_batch):
    default = tmp_path / "batches"
    custom = tmp_path / "results"
    kept = create_cli_batch("batch", default)
    removed = create_cli_batch("batch", custom)
    assert ensure_ids(custom) == {0: removed}

    shutil.rmtree(removed)
    assert main(["delete", "0", "--batches-dir", "results"]) == 0
    replacement = create_cli_batch("replacement", custom)
    assert ensure_ids(custom) == {0: replacement}
    assert ensure_ids(default) == {0: kept}


def test_delete_refuses_a_running_batch_and_an_unknown_id(tmp_path, create_cli_batch):
    directory = create_cli_batch("batch", tmp_path)
    assert ensure_ids(tmp_path) == {0: directory}
    command = [sys.executable, "-m", "expman", "delete", "--batches-dir", str(tmp_path)]
    with run_lock(directory):
        locked = subprocess.run([*command, "0"], capture_output=True, timeout=10)
        assert locked.returncode != 0
    unknown = subprocess.run([*command, "99"], capture_output=True, timeout=10)
    assert unknown.returncode != 0
    assert ensure_ids(tmp_path) == {0: directory}
    assert main(["delete", "0", "--batches-dir", str(tmp_path)]) == 0
    assert not directory.exists()


def test_failed_delete_reserves_the_id_until_a_retry_finishes(
    tmp_path, monkeypatch, create_cli_batch
):
    original = create_cli_batch("original", tmp_path)
    assert ensure_ids(tmp_path) == {0: original}
    command = ["delete", "0", "--batches-dir", str(tmp_path)]

    def fail_removal(path):
        raise PermissionError(f"cannot remove {path}")

    with monkeypatch.context() as failing:
        failing.setattr(shutil, "rmtree", fail_removal)
        with pytest.raises(PermissionError, match="cannot remove"):
            main(command)

    other = create_cli_batch("other", tmp_path)
    assert ensure_ids(tmp_path) == {1: other}
    assert main(command) == 0
    replacement = create_cli_batch("new", tmp_path)
    assert ensure_ids(tmp_path) == {0: replacement, 1: other}


def test_stop_without_a_reader_reports_and_cleans_up(tmp_path):
    output = tmp_path / "batch"
    output.mkdir()
    assert not request_stop(output)[0]

    endpoint = output / STOP_FIFO
    os.mkfifo(endpoint)

    assert not request_stop(output)[0]
    assert not endpoint.exists()


@pytest.mark.parametrize("symlink", [False, True])
def test_stop_control_preserves_unrelated_files(tmp_path, symlink):
    endpoint = tmp_path / STOP_FIFO
    contents = tmp_path / "ordinary" if symlink else endpoint
    contents.write_text("untouched")
    if symlink:
        endpoint.symlink_to(contents)
    assert not is_running(tmp_path)
    assert not request_stop(tmp_path)[0]
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
    for malformed in (":train", f"{pipeline_module}:"):
        with pytest.raises(SystemExit, match="module:attr"):
            main(["run", malformed, "unused.yaml"])
    with pytest.raises(SystemExit, match="could not import"):
        main(["run", "no_such_pipeline_module_xyz:train", "unused.yaml"])
    with pytest.raises(SystemExit, match="defines no attribute"):
        main(["run", f"{pipeline_module}:missing_attr", "unused.yaml"])
    with pytest.raises(SystemExit, match="must be a Pipeline"):
        main(["run", f"{pipeline_module}:not_a_pipeline", "unused.yaml"])


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


def test_resume_rejects_manifest_without_pipeline_location(tmp_path, create_cli_batch):
    store = BatchStore(create_cli_batch("batch", tmp_path))
    manifest = store.read_manifest()
    manifest["pipeline_spec"] = None
    store.save_manifest(manifest)
    with pytest.raises(expman.StorageError, match="no recorded pipeline location"):
        main(["resume", "0", "--batches-dir", str(tmp_path)])
