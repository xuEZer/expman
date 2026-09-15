import unittest
from unittest.mock import patch

from expman.randomness import RandomStateManager
from expman.storage import StorageError


class RandomStateManagerTests(unittest.TestCase):
    def setUp(self):
        self.manager = RandomStateManager(45)
        self.state = self.manager.capture()

    def test_cpu_only_snapshot_restores_when_cuda_becomes_visible(self):
        self.state["torch"]["cuda"] = []
        current_cpu_state = self.manager.torch.get_rng_state()

        with (
            patch.object(self.manager.torch.cuda, "is_available", return_value=True),
            patch.object(self.manager.torch.cuda, "device_count", return_value=1),
            patch.object(
                self.manager.torch.cuda,
                "get_rng_state_all",
                return_value=[current_cpu_state],
            ),
            patch.object(self.manager.torch.cuda, "set_rng_state_all") as set_cuda,
        ):
            self.manager.restore(self.state)

        set_cuda.assert_not_called()

    def test_nonempty_cuda_snapshot_must_match_visible_device_count(self):
        self.state["torch"]["cuda"] = [b"state"]

        with (
            patch.object(self.manager.torch.cuda, "is_available", return_value=True),
            patch.object(self.manager.torch.cuda, "device_count", return_value=2),
            self.assertRaisesRegex(
                StorageError,
                "saved CUDA random states do not match visible devices",
            ),
        ):
            self.manager.validate(self.state)


if __name__ == "__main__":
    unittest.main()
