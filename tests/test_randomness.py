import importlib.util
import os
import random
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from expman import (
    Batch,
    Experiment,
    Pipeline,
    RecoveryWarning,
    Stage,
    Status,
    StorageError,
    seed_everything,
)
from expman.randomness import RandomStateManager
from expman.storage import PickleSerializer, read_record, write_record


class Draw(Stage):
    def process(self, data, ctx):
        return [random.random() for _ in range(4)]


class CudaSamples(Stage):
    def process(self, data, ctx):
        import torch

        samples = ctx.state.setdefault("samples", [])
        while len(samples) < 3:
            samples.append(torch.rand(4, device="cuda:0").cpu().tolist())
            ctx.checkpoint.save(step=len(samples))
            if ctx.attempt == 1 and len(samples) == 1:
                torch.rand(4, device="cuda:0")
                torch.cuda.synchronize()
                os._exit(23)
        return samples


class RandomnessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        state = random.getstate()
        self.addCleanup(random.setstate, state)

    def test_default_root_seed_and_nested_seed(self):
        outputs = []
        for index, cfg in enumerate(({}, {"seed": 0}, {"model": {"seed": 900}})):
            result = Batch(
                Pipeline([Draw]), cfg, output_dir=self.root / str(index)
            ).run(progress=False)[0]
            outputs.append(result.output)
        expected = random.Random(0)
        self.assertEqual(outputs, [[expected.random() for _ in range(4)]] * 3)
        different = Experiment(
            Pipeline([Draw]), {"seed": 1}, output_dir=self.root / "other"
        ).run()
        self.assertNotEqual(outputs[0], different.output)

    def test_seed_validation_precedes_directory_creation(self):
        for seed in (None, True, -1, 2**32, "12", 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                Batch(Pipeline([]), {"seed": seed}, output_dir=self.root / "unused")
        self.assertFalse((self.root / "unused").exists())

    def test_public_seed_function_and_state_round_trip(self):
        seed_everything(42)
        expected = random.Random(42)
        self.assertEqual(random.random(), expected.random())
        manager = RandomStateManager(42)
        # Include the cached second Gaussian, not just uniform generator state.
        random.gauss(0, 1)
        snapshot = manager.capture()
        wanted = (random.random(), random.gauss(0, 1))
        manager.restore(snapshot)
        self.assertEqual((random.random(), random.gauss(0, 1)), wanted)

    def test_root_retry_without_checkpoint_reuses_initial_rng(self):
        draws = []

        class Work(Stage):
            def process(self, data, ctx):
                draws.append(random.random())
                if ctx.attempt == 1:
                    raise ValueError("retry")
                return draws[-1]

        with patch.object(
            RandomStateManager,
            "seed",
            autospec=True,
            side_effect=RandomStateManager.seed,
        ) as seeding:
            batch = Batch(
                Pipeline([Work]), {"seed": 11}, output_dir=self.root / "batch"
            )
            result = batch.run(progress=False)[0]
        self.assertEqual(result.status, Status.SUCCEEDED)
        self.assertEqual(draws, [random.Random(11).random()] * 2)
        self.assertEqual(seeding.call_count, 1)
        self.assertTrue((batch.experiments[0].output_dir / "rng_initial.pkl").exists())

    def test_completed_stage_restores_rng_before_failed_stage_reentry(self):
        observed = []

        class Prepare(Stage):
            def process(self, data, ctx):
                return random.random()

        class Work(Stage):
            def process(self, data, ctx):
                observed.append((data, random.random()))
                if ctx.attempt == 1:
                    raise ValueError("retry")
                return observed[-1]

        batch = Batch(
            Pipeline([Prepare, Work]), {"seed": 7}, output_dir=self.root / "batch"
        )
        batch.run(progress=False)
        expected = random.Random(7)
        self.assertEqual(observed, [(expected.random(), expected.random())] * 2)
        snapshot = read_record(
            batch.experiments[0]._store.stage_dir((0,)) / "completed.pkl",
            PickleSerializer(),
        )
        self.assertIn("rng_state", snapshot)
        self.assertEqual(snapshot["state"], {})

    def test_checkpoint_restores_after_constructor_and_before_process(self):
        observed = []

        class Work(Stage):
            def __init__(self):
                super().__init__()
                random.random()

            def process(self, data, ctx):
                if ctx.attempt == 1:
                    ctx.state["first"] = random.random()
                    ctx.checkpoint.save(step=0)
                    observed.append(random.random())
                    raise ValueError("retry")
                observed.append(random.random())
                return ctx.state

        batch = Batch(Pipeline([Work]), {"seed": 9}, output_dir=self.root / "batch")
        result = batch.run(progress=False)[0]
        self.assertEqual(observed[0], observed[1])
        self.assertEqual(set(result.output), {"first"})

    def test_corrupt_random_checkpoint_falls_back_without_advancing_rng(self):
        class Work(Stage):
            def process(self, data, ctx):
                if ctx.attempt == 1:
                    random.random()
                    ctx.state["step"] = 1
                    ctx.checkpoint.save(step=1)
                    random.random()
                    ctx.state["step"] = 2
                    path = ctx.checkpoint.save(step=2)
                    record = read_record(path, PickleSerializer())
                    record["rng_state"]["python"] = ("bad",)
                    write_record(path, record, PickleSerializer())
                    raise ValueError("retry")
                return ctx.state["step"], random.random()

        batch = Batch(Pipeline([Work]), {}, output_dir=self.root / "batch")
        with self.assertWarns(RecoveryWarning):
            result = batch.run(progress=False)[0]
        expected = random.Random(0)
        expected.random()
        self.assertEqual(result.output, (1, expected.random()))

    def test_random_capture_failure_is_a_stage_failure(self):
        capture = RandomStateManager.capture
        calls = []

        def fail_snapshot(manager):
            calls.append(True)
            if len(calls) == 2:
                raise StorageError("random capture failed")
            return capture(manager)

        batch = Batch(Pipeline([Draw]), {}, output_dir=self.root / "batch")
        with patch.object(RandomStateManager, "capture", fail_snapshot):
            result = batch.run(progress=False)[0]
        self.assertEqual(
            [item.status for item in result.attempts], [Status.FAILED, Status.SUCCEEDED]
        )
        self.assertEqual(result.attempts[0].error_type, "StorageError")
        expected = random.Random(0)
        self.assertEqual(result.output, [expected.random() for _ in range(4)])

    def test_legacy_snapshot_warns_when_random_state_is_missing(self):
        class Work(Stage):
            def process(self, data, ctx):
                if ctx.attempt == 1:
                    path = ctx.checkpoint.save()
                    record = read_record(path, PickleSerializer())
                    record.pop("rng_state")
                    write_record(path, record, PickleSerializer())
                    raise ValueError("retry")
                return 1

        batch = Batch(Pipeline([Work]), {}, output_dir=self.root / "batch")
        with self.assertWarnsRegex(RecoveryWarning, "legacy snapshot"):
            result = batch.run(progress=False)[0]
        self.assertEqual(result.status, Status.SUCCEEDED)

    def test_corrupt_initial_state_fails_instead_of_reseeding(self):
        experiment = Experiment(Pipeline([Draw]), {}, output_dir=self.root / "run")
        write_record(experiment.output_dir / "rng_initial.pkl", {}, PickleSerializer())
        result = experiment.run()
        self.assertEqual(result.status, Status.FAILED)
        self.assertEqual(result.error_type, "StorageError")

    def test_optional_absence_and_broken_import_are_distinguished(self):
        def missing(name):
            raise ModuleNotFoundError(name=name)

        with patch("expman.randomness.importlib.import_module", side_effect=missing):
            manager = RandomStateManager(0)
            manager.seed()
            self.assertIsNone(manager.capture()["numpy"])
            self.assertIsNone(manager.capture()["torch"])
        with (
            patch(
                "expman.randomness.importlib.import_module",
                side_effect=ModuleNotFoundError(name="broken_dependency"),
            ),
            self.assertRaises(ModuleNotFoundError),
        ):
            RandomStateManager(0)

    def test_saved_backend_cannot_be_silently_dropped(self):
        with patch("expman.randomness._optional_import", return_value=None):
            manager = RandomStateManager(0)
            snapshot = manager.capture()
            snapshot["numpy"] = ("MT19937", [], 0, 0, 0.0)
            with self.assertRaisesRegex(StorageError, "requires NumPy"):
                manager.restore(snapshot)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy optional")
    def test_numpy_global_state_including_gaussian_cache(self):
        import numpy as np

        manager = RandomStateManager(23)
        manager.seed()
        np.random.normal()
        snapshot = manager.capture()
        expected = (float(np.random.normal()), float(np.random.random()))
        manager.restore(snapshot)
        self.assertEqual(
            (float(np.random.normal()), float(np.random.random())), expected
        )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional")
    def test_torch_cpu_state(self):
        import torch

        manager = RandomStateManager(23)
        manager.seed()
        torch.rand(3)
        snapshot = manager.capture()
        expected = torch.rand(8)
        manager.restore(snapshot)
        self.assertTrue(torch.equal(torch.rand(8), expected))
        self.assertIsInstance(snapshot["torch"]["cpu"], bytes)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional")
    def test_cuda_worker_exit_restores_checkpoint_sequence(self):
        import torch

        if os.name != "posix" or not torch.cuda.is_available():
            self.skipTest("POSIX CUDA worker unavailable")
        from expman.devices import NvidiaMemory

        # Match the physical index to the currently visible logical GPU.
        uuid = torch.cuda.get_device_properties(0).uuid
        visible_uuid = (
            f"GPU-{UUID(bytes=bytes(uuid.bytes))}"
            if hasattr(uuid, "bytes")
            else str(uuid)
        )
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        indices = [
            int(line.split(",")[0])
            for line in query.stdout.splitlines()
            if line.split(",")[1].strip() == visible_uuid
        ]
        if not indices:
            self.skipTest("visible CUDA GPU is not a physical nvidia-smi device")
        device = indices[0]
        if NvidiaMemory([device]).sample()[device].free_ratio < 0.25:
            self.skipTest("insufficient free GPU memory for integration test")
        reference = torch.Generator(device="cuda:0").manual_seed(23)
        expected = [
            torch.rand(4, device="cuda:0", generator=reference).cpu().tolist()
            for _ in range(3)
        ]
        batch = Batch(
            Pipeline([CudaSamples]),
            {"seed": 23, "device": [device]},
            output_dir=self.root / "cuda-batch",
        )
        result = batch.run(progress=False)[0]
        self.assertEqual(result.status, Status.SUCCEEDED)
        self.assertEqual(result.output, expected)
        self.assertEqual(
            [item.status for item in result.attempts], [Status.FAILED, Status.SUCCEEDED]
        )
        self.assertEqual(result.attempts[0].error_type, "WorkerExit")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch optional")
    def test_torch_cuda_state(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        manager = RandomStateManager(23)
        manager.seed()
        snapshot = manager.capture()
        expected = [
            torch.rand(4, device=f"cuda:{index}")
            for index in range(torch.cuda.device_count())
        ]
        manager.restore(snapshot)
        for index, values in enumerate(expected):
            self.assertTrue(torch.equal(torch.rand(4, device=f"cuda:{index}"), values))


if __name__ == "__main__":
    unittest.main()
