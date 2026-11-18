"""End-to-end CLI tests on a reserved physical CUDA/MPS device."""

import os
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep

import pytest

from expman import Batch, Pipeline, Stage, Status


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
        return ctx.checkpoint_state["__payload__"]


cuda_pipeline = Pipeline([SumSquares, Scale])


class PauseAfterCheckpoint(Stage):
    def process(self, data, ctx):
        import torch

        total = ctx.checkpoint_state or 0
        for step in ctx.resume_range(0, 2):
            total += torch.tensor(step + 10, device="cuda").item()
            ctx.checkpoint(step + 1, total)
            if step == 0:
                Path(ctx.cfg["ready"]).touch()
                sleep(120)
        return total


pause_pipeline = Pipeline([PauseAfterCheckpoint])
CLI = [sys.executable, "-m", "expman"]


@pytest.fixture
def cli_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parent), *sys.path]
    )
    return env


def test_cuda_batch_reuses_prefix_and_recovers_checkpoint(
    tmp_path, cli_env, gpu_device
):
    config = tmp_path / "experiment.yaml"
    config.write_text(
        f"device: [{gpu_device}]\nseed: 42\nfactor: 2\ngrid:\n  factor: [2, 3]\n"
    )
    directory = tmp_path / "batch"
    subprocess.run(
        [
            *CLI,
            "run",
            f"{__name__}:cuda_pipeline",
            str(config),
            "-o",
            str(directory),
        ],
        cwd=tmp_path,
        env=cli_env,
        check=True,
        timeout=150,
    )
    batch = Batch.resume(cuda_pipeline, directory)
    results = batch.results

    assert all(result.status == Status.SUCCEEDED for result in results), results
    expected = [sum(i * i for i in range(256)) * factor for factor in (2, 3)]
    assert [result.output for result in results] == expected
    for result in results:
        assert [a.error_message for a in result.attempts] == [
            "retry from checkpoint",
            None,
        ]

    history = batch.state.stage_history
    assert sum(row["reused"] for row in history if row["stage"] == 0) == 1
    workers = [row for row in history if not row["reused"]]
    assert all(
        row["gpu_limit_verified"] and row["device"] == gpu_device for row in workers
    )
    for result in results:
        measured = sum(
            row["duration"] for row in workers if row["run_id"] == result.run_id
        )
        assert result.duration_seconds == pytest.approx(measured)


def test_stop_and_resume_from_a_committed_cuda_checkpoint(
    tmp_path, cli_env, gpu_device
):
    ready = tmp_path / "first-checkpoint"
    config = tmp_path / "experiment.yaml"
    config.write_text(f"device: [{gpu_device}]\nseed: 42\nready: {ready}\n")
    directory = tmp_path / "batch"
    where = {"cwd": tmp_path, "env": cli_env}
    owner = subprocess.Popen(
        [*CLI, "run", f"{__name__}:pause_pipeline", str(config), "-o", str(directory)],
        **where,
    )
    try:
        deadline = monotonic() + 60
        while not ready.exists():
            assert owner.poll() is None, "the batch exited before its checkpoint"
            assert monotonic() < deadline, "the worker did not publish a checkpoint"
            sleep(0.1)
        subprocess.run([*CLI, "stop", str(directory)], **where, check=True, timeout=15)
        assert owner.wait(timeout=30) == 143
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=10)

    subprocess.run([*CLI, "resume", str(directory)], **where, check=True, timeout=90)
    batch = Batch.resume(pause_pipeline, directory)
    (result,) = batch.results
    assert [attempt.status for attempt in result.attempts] == [
        Status.CANCELLED,
        Status.SUCCEEDED,
    ]
    assert result.output == 21
    assert [row["attempt"] for row in batch.state.stage_history] == [1, 2]
    assert all(
        row["device"] == gpu_device and row["gpu_limit_verified"]
        for row in batch.state.stage_history
    )
