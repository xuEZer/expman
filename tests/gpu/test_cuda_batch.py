"""End-to-end CLI tests on a reserved physical CUDA/MPS device."""

import os
import shlex
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep

import pytest

from expman import Pipeline, Stage
from expman.observation.registry import ensure_ids
from expman.observation.status import RecordedBatch
from expman.storage.batch import BatchStore
from expman.storage.prefix_cache import PrefixCache
from expman.storage.run import RunStore


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


def read_outputs(directory, batch):
    """Resolve each run's final stage snapshot, including shared prefixes."""
    store = BatchStore(directory)
    signature = batch.manifest["pipeline"]
    outputs = []
    for run in batch.runs:
        seed = store.load_config(run.run_id)["seed"]
        snapshots = RunStore(directory / "experiments" / run.run_id)
        snapshots.shared = PrefixCache(
            directory / "cache", signature, seed, store.serializer
        )
        final = snapshots.completed((len(signature) - 1,))
        assert final is not None
        outputs.append(final["output"])
    return outputs


def test_cuda_batch_reuses_prefix_and_recovers_checkpoint(
    tmp_path, cli_env, gpu_device
):
    config = tmp_path / "experiment.yaml"
    config.write_text(
        f"device: [{gpu_device}]\nseed: 42\nfactor: 2\ngrid:\n  factor: [2, 3]\n"
    )
    subprocess.run(
        [
            *CLI,
            "run",
            f"{__name__}:cuda_pipeline",
            str(config),
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
    assert read_outputs(directory, batch) == expected
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
    config = tmp_path / "experiment.yaml"
    config.write_text(f"device: [{gpu_device}]\nseed: 42\nready: {ready}\n")
    parent = tmp_path / "training batches"
    where = {"cwd": tmp_path, "env": cli_env}
    owner = subprocess.Popen(
        [
            *CLI,
            "run",
            f"{__name__}:pause_pipeline",
            str(config),
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
    subprocess.run([*CLI, *resume[1:]], **where, check=True, timeout=90)
    directory = ensure_ids(parent)[0]
    batch = RecordedBatch.read(directory)
    (run,) = batch.runs
    assert [attempt["status"] for attempt in run.attempts] == ["cancelled", "succeeded"]
    assert read_outputs(directory, batch) == [21]
    history = batch.manifest["stage_history"]
    assert [row["attempt"] for row in history] == [1, 2]
    assert all(
        row["device"] == gpu_device and row["gpu_limit_verified"] for row in history
    )
