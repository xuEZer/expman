"""Global RNG initialization and portable state snapshots for experiment recovery."""

import importlib
import random
from copy import deepcopy

from .storage import StorageError


def validate_seed(seed: int) -> int:
    """Use the common unsigned 32-bit seed range supported by all backends."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    return seed


def _optional_import(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        return None


class RandomStateManager:
    def __init__(self, seed: int):
        self.seed_value = validate_seed(seed)
        self.numpy = _optional_import("numpy")
        self.torch = _optional_import("torch")

    def seed(self) -> None:
        random.seed(self.seed_value)
        if self.numpy is not None:
            self.numpy.random.seed(self.seed_value)
        if self.torch is not None:
            self.torch.manual_seed(self.seed_value)

    def capture(self) -> dict:
        try:
            state = {
                "version": 1,
                "seed": self.seed_value,
                "python": random.getstate(),
                "numpy": None,
                "torch": None,
            }
            if self.numpy is not None:
                algorithm, keys, position, gaussian, cached = (
                    self.numpy.random.get_state()
                )
                state["numpy"] = (algorithm, keys.tolist(), position, gaussian, cached)
            if self.torch is not None:
                state["torch"] = {
                    "cpu": bytes(self.torch.get_rng_state().tolist()),
                    "cuda": [
                        bytes(item.tolist())
                        for item in self.torch.cuda.get_rng_state_all()
                    ]
                    if self.torch.cuda.is_available()
                    else [],
                }
            return deepcopy(state)
        except Exception as error:
            raise StorageError(f"could not capture random state: {error}") from error

    def validate(self, state: dict) -> None:
        try:
            if (
                not isinstance(state, dict)
                or state.get("version") != 1
                or state.get("seed") != self.seed_value
                or not {"python", "numpy", "torch"} <= state.keys()
            ):
                raise ValueError("invalid random-state version or seed")
            random.Random(0).setstate(state["python"])
            if state["numpy"] is not None:
                if self.numpy is None:
                    raise ValueError("saved random state requires NumPy")
                self.numpy.random.RandomState(0).set_state(state["numpy"])
            if state["torch"] is not None:
                if self.torch is None:
                    raise ValueError("saved random state requires PyTorch")
                cpu = state["torch"]["cpu"]
                cuda = state["torch"]["cuda"]
                if not isinstance(cpu, bytes) or not isinstance(cuda, list):
                    raise ValueError("invalid PyTorch random state")
                self.torch.Generator(device="cpu").set_state(self._tensor(cpu))
                count = (
                    self.torch.cuda.device_count()
                    if self.torch.cuda.is_available()
                    else 0
                )
                if len(cuda) != count or any(
                    not isinstance(item, bytes) or not item for item in cuda
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
        if state["numpy"] is not None:
            self.numpy.random.set_state(state["numpy"])
        if state["torch"] is not None:
            self.torch.set_rng_state(self._tensor(state["torch"]["cpu"]))
            if state["torch"]["cuda"]:
                self.torch.cuda.set_rng_state_all(
                    [self._tensor(item) for item in state["torch"]["cuda"]]
                )

    def restore(self, state: dict) -> None:
        self.validate(state)
        previous = self.capture()
        try:
            self._apply(state)
        except Exception as error:
            self._apply(previous)
            raise StorageError(f"could not restore random state: {error}") from error


def seed_everything(seed: int = 0) -> None:
    """Seed Python and installed NumPy/PyTorch global generators.

    Independent generators, data-loader workers and deterministic GPU algorithms
    remain the responsibility of user code. Optional library import errors propagate.
    """
    RandomStateManager(seed).seed()
