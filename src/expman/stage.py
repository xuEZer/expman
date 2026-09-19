"""The user extension point for pipeline data processing."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Generic, TypeVar, final

from .context import RunContext, _name

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class Stage(ABC, Generic[InputT, OutputT]):
    """Override process and use run as the monitored execution entry point.

    Stage instances may hold state. Pipeline constructs them with no arguments
    on each execution; read the current experiment configuration from ctx.cfg.
    Calling an instance directly through run preserves that instance's state.
    """

    def __init__(self, *, name: str | None = None) -> None:
        self._name = type(self).__name__ if name is None else name
        _name(self._name, "stage name")

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def config_dependencies(cls, cfg: Mapping[str, Any]) -> Mapping | bool:
        """Declare which configuration positions this Stage result depends on.

        The declaration mirrors the configuration tree:

        - ``True`` at a node depends on the whole subtree at that position;
        - a mapping recurses into its keys (strings for mappings, integers for
          sequence indices);
        - ``False`` or an absent key declares no dependency.

        ``cfg`` is the read-only configuration of the current attempt, so a
        branch may declare different dependencies per run. Returning ``True``
        declares the entire configuration. The default is conservative: a Stage
        that reads ``ctx.cfg`` without overriding this method never reuses stale
        results.
        """
        return True

    @final
    def run(self, data: InputT, ctx: RunContext | None = None) -> OutputT:
        """Execute process once, with stage-level timing and outcome events."""
        context = RunContext() if ctx is None else ctx
        with context.observe(self.name, kind="stage") as stage_context:
            output = self.process(data, stage_context)
            if context._store is not None:
                checkpoint = context._checkpoint
                context._store.complete(
                    context._stage_path,
                    output,
                    context.state,
                    self.name,
                    dependencies=None
                    if checkpoint is None
                    else checkpoint.dependencies,
                )
            return output

    @abstractmethod
    def process(self, data: InputT, ctx: RunContext) -> OutputT:
        """Transform input into output, or raise an exception on failure."""
        ...
