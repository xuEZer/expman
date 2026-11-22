import random

import pytest

from expman.core.randomness import RandomStateManager
from expman.storage.base import StorageError


def test_snapshot_without_cuda_replays_all_cpu_generators():
    manager = RandomStateManager(45)
    state = manager.capture()
    state["torch"]["cuda"] = []

    def draw():
        return (
            random.random(),
            manager.numpy.random.random(),
            manager.torch.rand(()).item(),
        )

    expected = draw()
    before_cuda = manager.capture()["torch"]["cuda"]
    manager.restore(state)
    assert draw() == expected
    assert manager.capture()["torch"]["cuda"] == before_cuda


def test_nonempty_cuda_snapshot_must_match_visible_device_count():
    manager = RandomStateManager(45)
    state = manager.capture()
    state["torch"]["cuda"] = [b"state"] * (manager.torch.cuda.device_count() + 1)
    with pytest.raises(StorageError, match="do not match visible devices"):
        manager.validate(state)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_incomplete_snapshot_preserves_random_state(backend):
    manager = RandomStateManager(45)
    previous = manager.capture()
    advanced = random.Random()
    advanced.setstate(previous["python"])
    advanced.random()
    incomplete = {**previous, "python": advanced.getstate(), backend: None}
    with pytest.raises(StorageError, match="could not validate random state"):
        manager.restore(incomplete)
    assert manager.capture() == previous
