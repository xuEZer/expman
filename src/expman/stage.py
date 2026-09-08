"""The user extension point for pipeline data processing."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, final

from .context import RunContext, _name

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Stage(ABC, Generic[InputT, OutputT]):
    """Override process and use run as the monitored execution entry point.

    Stage instances may hold state. Reusing an instance preserves that state;
    callers should create a fresh instance when an independent run is needed.
    """

    def __init__(self, *, name: str | None = None) -> None:
        self._name = type(self).__name__ if name is None else name
        _name(self._name, "stage name")

    @property
    def name(self) -> str:
        return self._name

    @final
    def run(self, data: InputT, ctx: RunContext | None = None) -> OutputT:
        """Execute process once, with stage-level timing and outcome events."""
        context = RunContext() if ctx is None else ctx
        with context.observe(self.name, kind="stage") as stage_context:
            return self.process(data, stage_context)

    @abstractmethod
    def process(self, data: InputT, ctx: RunContext) -> OutputT:
        """Transform input into output, or raise an exception on failure."""
        ...
