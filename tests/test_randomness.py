from unittest.mock import patch

import pytest

from expman.randomness import RandomStateManager
from expman.storage import StorageError


@pytest.fixture
def manager():
    return RandomStateManager(45)


@pytest.fixture
def state(manager):
    return manager.capture()


def test_cpu_only_snapshot_restores_when_cuda_becomes_visible(manager, state):
    state["torch"]["cuda"] = []
    current_cpu_state = manager.torch.get_rng_state()

    with (
        patch.object(manager.torch.cuda, "is_available", return_value=True),
        patch.object(manager.torch.cuda, "device_count", return_value=1),
        patch.object(
            manager.torch.cuda,
            "get_rng_state_all",
            return_value=[current_cpu_state],
        ),
        patch.object(manager.torch.cuda, "set_rng_state_all") as set_cuda,
    ):
        manager.restore(state)

    set_cuda.assert_not_called()


def test_nonempty_cuda_snapshot_must_match_visible_device_count(manager, state):
    state["torch"]["cuda"] = [b"state"]

    with (
        patch.object(manager.torch.cuda, "is_available", return_value=True),
        patch.object(manager.torch.cuda, "device_count", return_value=2),
        pytest.raises(
            StorageError,
            match="saved CUDA random states do not match visible devices",
        ),
    ):
        manager.validate(state)
