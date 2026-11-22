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

MODULE_NAME = "src.pipeline"

MODULE_SOURCE = """\
from expman import Pipeline, Stage


class Train(Stage):
    def process(self, data, ctx):
        if ctx.cfg["fail"]:
            raise RuntimeError("training failed")


train = Pipeline([Train])
empty = Pipeline([])
not_a_pipeline = 42
"""

SETTINGS_MODULE_NAME = "cli_pipeline_settings"

SETTINGS_MODULE_SOURCE = """\
import expman

expman.set(retries=3, coverage=0.9, batches_dir="pipeline-batches")

train = expman.Pipeline([])
"""


@pytest.fixture()
def pipeline_module(tmp_path, monkeypatch):
    """Expose the default pipeline module from an isolated experiment project."""
    source = tmp_path / "src"
    source.mkdir()
    (source / "pipeline.py").write_text(MODULE_SOURCE)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "src", raising=False)
    monkeypatch.delitem(sys.modules, MODULE_NAME, raising=False)
    try:
        yield MODULE_NAME
    finally:
        sys.modules.pop(MODULE_NAME, None)
        sys.modules.pop("src", None)


@pytest.fixture
def prepare_batch_only(monkeypatch):
    """Let CLI commands persist a real batch without starting its worker processes."""
    prepared = []

    def capture(batch, **options):
        prepared.append(batch)
        return 0

    monkeypatch.setattr(cli, "_execute", capture)
    return prepared


def write_config(project, name="baseline", *, fail=False):
    directory = project / "configs" / "experiments"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.yaml").write_text(
        f"device: [0]\nseed: 0\nfail: {str(fail).lower()}\n"
    )
    return name


@pytest.fixture
def create_cli_batch(tmp_path, pipeline_module, prepare_batch_only):
    def create(name, root):
        config = write_config(tmp_path, name)
        assert (
            main(
                [
                    "run",
                    f"{pipeline_module}:empty",
                    config,
                    "--batches-dir",
                    str(root),
                ]
            )
            == 0
        )
        return prepare_batch_only[-1].output_dir

    return create


def test_run_creates_a_named_batch_under_the_output_parent(
    tmp_path, pipeline_module, prepare_batch_only, monkeypatch
):
    monkeypatch.setattr(cli, "strftime", datetime(2026, 1, 2, 3, 4, 5).strftime)
    config = write_config(tmp_path, "experiment.large")
    (tmp_path / "configs" / f"{config}.yaml").write_text("invalid: config root copy")
    command = ["run", f"{pipeline_module}:train", config]

    assert main(command) == 0
    parent = tmp_path / "batches"
    directory = parent / "train_experiment.large_20260102030405"
    manifest = directory / "batch.pkl"
    original = manifest.read_bytes()
    with pytest.raises(FileExistsError) as collision:
        main(command)
    assert collision.value.filename == str(directory)
    assert manifest.read_bytes() == original
    assert ensure_ids(parent) == {0: directory}


def test_run_requires_a_config_name(tmp_path, pipeline_module, capsys):
    name = write_config(tmp_path)
    for argument in (
        str(tmp_path / "configs" / "experiments" / name),
        f"configs/experiments/{name}",
        f"../{name}",
        f"{name}.yaml",
        f"{name}.yml",
    ):
        with pytest.raises(SystemExit):
            main(["run", f"{pipeline_module}:empty", argument])
        assert "expected a config name" in capsys.readouterr().err
    assert not (tmp_path / "batches").exists()


def test_run_requires_a_yaml_file_in_experiments(tmp_path, pipeline_module):
    name = write_config(tmp_path)
    config = tmp_path / "configs" / "experiments" / f"{name}.yaml"
    shutil.copyfile(config, tmp_path / "configs" / config.name)
    config.rename(config.with_suffix(".yml"))
    with pytest.raises(expman.ConfigError, match=r"configs/experiments/baseline\.yaml"):
        main(["run", f"{pipeline_module}:empty", name])


def test_run_rejects_a_config_symlink_outside_experiments(
    tmp_path, pipeline_module, capsys
):
    name = write_config(tmp_path)
    config = tmp_path / "configs" / "experiments" / f"{name}.yaml"
    outside = config.rename(tmp_path / "configs" / config.name)
    config.symlink_to(outside)
    with pytest.raises(SystemExit):
        main(["run", f"{pipeline_module}:empty", name])
    assert "points outside ./configs/experiments/" in capsys.readouterr().err
    assert not (tmp_path / "batches").exists()


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
                    write_config(tmp_path),
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


def test_resume_of_a_short_pipeline_location_uses_the_builtin_module(
    tmp_path, pipeline_module, prepare_batch_only
):
    assert main(["run", "train", write_config(tmp_path)]) == 0
    store = BatchStore(ensure_ids(tmp_path / "batches")[0])
    manifest = store.read_manifest()
    assert manifest["pipeline_spec"] == "src.pipeline:train"
    manifest["pipeline_spec"] = "train"
    store.save_manifest(manifest)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.expman]\npipeline_module = "different"\n'
    )
    assert main(["resume", "0"]) == 0


def test_project_defaults_apply_across_cli_processes_and_explicit_arguments_win(
    tmp_path,
):
    config = write_config(tmp_path)
    project = tmp_path / "pyproject.toml"
    project.write_text(
        '[tool.expman]\npipeline_module = "workflows"\nbatches_dir = "project-batches"\n'
    )
    (tmp_path / "workflows.py").write_text(
        "from expman import Pipeline\ntrain = Pipeline([])\n"
    )
    cli = [sys.executable, "-m", "expman"]
    for options, parent in [
        ([], "project-batches"),
        (["--batches-dir", "custom"], "custom"),
    ]:
        stdout = subprocess.check_output(
            [*cli, "run", "train", config, *options], cwd=tmp_path, text=True
        )
        directory = ensure_ids(tmp_path / parent)[0]
        store = BatchStore(directory)
        assert "Batch ID: 0" in stdout and f"Batch directory: {directory}" in stdout
        assert ": succeeded" in stdout
        assert store.read_manifest()["pipeline_spec"] == "workflows:train"

    project.write_text(project.read_text().replace('"workflows"', '"different"'))
    store = BatchStore(ensure_ids(tmp_path / "project-batches")[0])
    experiments = store.read_manifest()["experiments"]
    stdout = subprocess.check_output([*cli, "resume", "0"], cwd=tmp_path, text=True)
    assert f"{experiments[0]['run_id']}: succeeded" in stdout
    assert store.read_manifest()["experiments"] == experiments


@pytest.mark.simulated_gpu
def test_run_and_resume_report_terminal_failure_to_the_shell(
    tmp_path, pipeline_module, capsys
):
    (tmp_path / "pyproject.toml").write_text("[tool.expman]\nretries = 0\n")
    config = write_config(tmp_path, "failing", fail=True)
    assert (
        main(
            [
                "run",
                f"{pipeline_module}:train",
                config,
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
                write_config(tmp_path),
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
    config = write_config(tmp_path)
    assert main(["run", f"{pipeline_module}:empty", config]) == 0
    options = ["--batches-dir", "other"]
    assert main(["run", f"{pipeline_module}:train", config, *options]) == 0
    batch = prepare_batch_only[-1]
    run_id = batch.experiments[0].run_id

    def start_run():
        batch.state.start(run_id, device=0, stage_index=0)
        batch._save()

    first, second = watch_status("0", *options, after_first_frame=start_run)
    assert f"{run_id}  pending" in first
    assert f"{run_id}  running" in second
    for frame in (first, second):
        assert str(batch.output_dir) in frame
        assert "stage 1/1" in frame
        assert "elapsed" in frame


def test_status_overview_lists_batches_with_ids(
    tmp_path, pipeline_module, prepare_batch_only, watch_status
):
    config = write_config(tmp_path)
    main(["run", f"{pipeline_module}:train", config])
    first, second = watch_status(
        after_first_frame=lambda: main(["run", f"{pipeline_module}:empty", config]),
    )
    ids = ensure_ids(tmp_path / "batches")
    assert f" 0  batches/{ids[0].name} " in first
    assert ids[1].name not in first
    assert f" 1  batches/{ids[1].name} " in second


@pytest.mark.parametrize("command", ["status", "resume", "stop", "delete"])
def test_batch_ids_reject_directory_paths(command):
    with pytest.raises(SystemExit) as error:
        main([command, "batches/train_experiments"])
    assert error.value.code == 2


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
        locked = subprocess.run(
            [*command, "0"], capture_output=True, text=True, timeout=10
        )
        assert locked.returncode != 0
        assert "already running" in locked.stderr
    with pytest.raises(expman.StorageError, match="unknown batch ID 99"):
        main(["delete", "99", "--batches-dir", str(tmp_path)])
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


def test_stop_without_a_reader_reports_and_cleans_up(tmp_path, create_cli_batch):
    output = create_cli_batch("batch", tmp_path / "batches")
    assert main(["stop", "0"]) == 1
    with pytest.raises(SystemExit, match="unknown batch ID -1"):
        main(["stop", "-1"])

    endpoint = output / STOP_FIFO
    os.mkfifo(endpoint)

    assert main(["stop", "0"]) == 1
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
            main(["run", malformed, "unused"])
    with pytest.raises(SystemExit, match="could not import"):
        main(["run", "no_such_pipeline_module_xyz:train", "unused"])
    with pytest.raises(SystemExit, match="defines no attribute"):
        main(["run", f"{pipeline_module}:missing_attr", "unused"])
    with pytest.raises(SystemExit, match="must be a Pipeline"):
        main(["run", f"{pipeline_module}:not_a_pipeline", "unused"])


def test_resume_rejects_manifest_without_pipeline_location(tmp_path, create_cli_batch):
    store = BatchStore(create_cli_batch("batch", tmp_path))
    manifest = store.read_manifest()
    manifest["pipeline_spec"] = None
    store.save_manifest(manifest)
    with pytest.raises(expman.StorageError, match="no recorded pipeline location"):
        main(["resume", "0", "--batches-dir", str(tmp_path)])
