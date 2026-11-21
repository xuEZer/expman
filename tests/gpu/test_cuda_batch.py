"""End-to-end CLI tests on a reserved physical CUDA/MPS device."""

import os
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep

import pytest

from expman import Pipeline, Stage
from expman.observation.registry import ensure_ids
from expman.observation.status import RecordedBatch


class SumSquares(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return False

    def process(self, data, ctx):
        import torch

        values = torch.arange(256, device="cuda")
        return values.square().sum().item()


class Scale(Stage):
    def process(self, data, ctx):
        import torch

        for step in ctx.resume_range(0, 1):
            product = torch.tensor(data, device="cuda") * ctx.cfg["factor"]
            ctx.checkpoint(step + 1, {"__payload__": product.item()})
            raise RuntimeError("retry from checkpoint")
        ctx.log_metrics({"total": ctx.checkpoint_state["__payload__"]})


cuda_pipeline = Pipeline([SumSquares, Scale])


class Increment(Stage):
    def process(self, data, ctx):
        return data + 1


class PauseAfterCheckpoint(Stage):
    def process(self, data, ctx):
        import torch

        total = data if ctx.checkpoint_state is None else ctx.checkpoint_state
        for step in ctx.resume_range(0, 2):
            total += torch.tensor(step + 10, device="cuda").item()
            ctx.checkpoint(step + 1, total)
            if step == 0:
                Path(ctx.cfg["ready"]).touch()
                sleep(120)
        ctx.log_metrics({"total": total})


pause_pipeline = Pipeline([SumSquares, Increment, PauseAfterCheckpoint])
CLI = [sys.executable, "-m", "expman"]


@pytest.fixture
def cli_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parent), *sys.path]
    )
    return env


def recorded_totals(directory, batch):
    with sqlite3.connect(directory / "metrics.sqlite3") as db:
        values = dict(
            db.execute(
                "SELECT run_id, value FROM metrics WHERE metric_path = ? AND step = -1",
                ('["total"]',),
            )
        )
    return [values[run.run_id] for run in batch.runs]


def test_cuda_batch_reuses_prefix_and_recovers_checkpoint(
    tmp_path, cli_env, gpu_device
):
    config = tmp_path / "configs" / "experiment.yaml"
    config.parent.mkdir()
    config.write_text(
        f"device: [{gpu_device}]\nseed: 42\nfactor: 2\ngrid:\n  factor: [2, 3]\n"
    )
    subprocess.run(
        [
            *CLI,
            "run",
            f"{__name__}:cuda_pipeline",
            config.stem,
            "--batches-dir",
            str(tmp_path),
        ],
        cwd=tmp_path,
        env=cli_env,
        check=True,
        timeout=150,
    )
    directory = ensure_ids(tmp_path)[0]
    batch = RecordedBatch.read(directory)
    assert [run.state for run in batch.runs] == ["succeeded", "succeeded"]
    expected = [sum(i * i for i in range(256)) * factor for factor in (2, 3)]
    assert recorded_totals(directory, batch) == expected
    assert not list((directory / "cache").rglob("*.pkl"))
    for run in batch.runs:
        assert [attempt["error_message"] for attempt in run.attempts] == [
            "retry from checkpoint",
            None,
        ]

    history = batch.manifest["stage_history"]
    assert sum(row["reused"] for row in history if row["stage"] == 0) == 1
    workers = [row for row in history if not row["reused"]]
    assert all(
        row["gpu_limit_verified"] and row["device"] == gpu_device for row in workers
    )
    for run in batch.runs:
        measured = sum(
            row["duration"] for row in workers if row["run_id"] == run.run_id
        )
        elapsed = sum(attempt["duration_seconds"] for attempt in run.attempts)
        assert elapsed == pytest.approx(measured)


def test_stop_and_resume_from_a_committed_cuda_checkpoint(
    tmp_path, cli_env, gpu_device
):
    ready = tmp_path / "first-checkpoint"
    config = tmp_path / "configs" / "experiment.yaml"
    config.parent.mkdir()
    config.write_text(f"device: [{gpu_device}]\nseed: 42\nready: {ready}\n")
    parent = tmp_path / "training batches"
    where = {"cwd": tmp_path, "env": cli_env}
    owner = subprocess.Popen(
        [
            *CLI,
            "run",
            f"{__name__}:pause_pipeline",
            config.stem,
            "--batches-dir",
            str(parent),
        ],
        stderr=subprocess.PIPE,
        text=True,
        **where,
    )
    try:
        deadline = monotonic() + 60
        while not ready.exists():
            assert owner.poll() is None, "the batch exited before its checkpoint"
            assert monotonic() < deadline, "the worker did not publish a checkpoint"
            sleep(0.1)
        subprocess.run(
            [*CLI, "stop", "0", "--batches-dir", str(parent)],
            **where,
            check=True,
            timeout=15,
        )
        _, stderr = owner.communicate(timeout=30)
        assert owner.returncode == 143
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=10)

    hint = next(
        line
        for line in stderr.splitlines()
        if line.startswith("Stopped; resume with: ")
    )
    resume = shlex.split(hint.removeprefix("Stopped; resume with: "))
    assert resume == ["expman", "resume", "0", "--batches-dir", str(parent)]
    directory = ensure_ids(parent)[0]
    cache = directory / "cache"
    assert not list(cache.glob("*/root/*/*.pkl"))
    assert len(list(cache.rglob("snapshot.pkl"))) == 1

    subprocess.run([*CLI, *resume[1:]], **where, check=True, timeout=90)
    batch = RecordedBatch.read(directory)
    (run,) = batch.runs
    assert [attempt["status"] for attempt in run.attempts] == ["cancelled", "succeeded"]
    assert recorded_totals(directory, batch) == [
        sum(i * i for i in range(256)) + 1 + 21
    ]
    assert not list(cache.rglob("*.pkl"))
    history = batch.manifest["stage_history"]
    assert [(row["stage"], row["attempt"]) for row in history] == [
        (0, 1),
        (1, 1),
        (2, 1),
        (2, 2),
    ]
    assert all(
        row["device"] == gpu_device and row["gpu_limit_verified"] for row in history
    )
