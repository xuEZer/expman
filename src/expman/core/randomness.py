"""Global RNG initialization and portable state snapshots for experiment recovery."""

import random
from importlib import import_module

from expman.storage.base import StorageError


def validate_seed(seed: int) -> int:
    """Use the common unsigned 32-bit seed range supported by all backends."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    return seed


class RandomStateManager:
    def __init__(self, seed: int):
        self.seed_value = validate_seed(seed)
        # Workers verify their MPS client before importing either backend.
        self.numpy = import_module("numpy")
        self.torch = import_module("torch")

    def seed(self) -> None:
        random.seed(self.seed_value)
        self.numpy.random.seed(self.seed_value)
        self.torch.manual_seed(self.seed_value)

    def capture(self) -> dict:
        try:
            algorithm, keys, position, gaussian, cached = self.numpy.random.get_state()
            return {
                "version": 1,
                "seed": self.seed_value,
                "python": random.getstate(),
                "numpy": (algorithm, keys.tolist(), position, gaussian, cached),
                "torch": {
                    "cpu": bytes(self.torch.get_rng_state().tolist()),
                    "cuda": [
                        bytes(item.tolist())
                        for item in self.torch.cuda.get_rng_state_all()
                    ]
                    if self.torch.cuda.is_available()
                    else [],
                },
            }
        except Exception as error:
            raise StorageError(f"could not capture random state: {error}") from error

    def validate(self, state: dict) -> None:
        try:
            if (
                not isinstance(state, dict)
                or state.get("version") != 1
                or state.get("seed") != self.seed_value
            ):
                raise ValueError("invalid random-state version or seed")
            random.Random(0).setstate(state["python"])
            self.numpy.random.RandomState(0).set_state(state["numpy"])
            cpu, cuda = state["torch"]["cpu"], state["torch"]["cuda"]
            if not isinstance(cpu, bytes) or not isinstance(cuda, list):
                raise ValueError("invalid PyTorch random state")
            self.torch.Generator(device="cpu").set_state(self._tensor(cpu))
            # A snapshot without CUDA streams can restore the CPU generators
            # even when a later attempt runs on a GPU.
            if cuda and (
                len(cuda) != self.torch.cuda.device_count()
                or any(not isinstance(item, bytes) or not item for item in cuda)
            ):
                raise ValueError(
                    "saved CUDA random states do not match visible devices"
                )
        except Exception as error:
            raise StorageError(f"could not validate random state: {error}") from error

    def _tensor(self, data):
        return self.torch.tensor(list(data), dtype=self.torch.uint8, device="cpu")

    def _apply(self, state):
        random.setstate(state["python"])
        self.numpy.random.set_state(state["numpy"])
        self.torch.set_rng_state(self._tensor(state["torch"]["cpu"]))
        cuda = state["torch"]["cuda"]
        if cuda:
            self.torch.cuda.set_rng_state_all([self._tensor(item) for item in cuda])

    def restore(self, state: dict) -> None:
        self.validate(state)
        previous = self.capture()
        try:
            self._apply(state)
        except Exception as error:
            self._apply(previous)
            raise StorageError(f"could not restore random state: {error}") from error


def seed_everything(seed: int) -> None:
    """Seed the global Python, NumPy and PyTorch generators."""
    RandomStateManager(seed).seed()
