"""The user extension point for pipeline data processing."""

from collections.abc import Mapping
from typing import Any, final

from expman.core.context import RunContext, _name
from expman.core.execution import stage_scope


class Stage[InputT, OutputT]:
    """Implement process() to compute a Stage output.

    A Stage is a single data transformation. The user owns the whole execution:
    loop control, checkpoint timing and state shape are explicit through
    ctx.resume_range() and ctx.checkpoint() during managed execution. Pipeline
    constructs a fresh instance when it needs to run process(). Read the
    current experiment configuration from ctx.cfg.

    State carried through a resume is whatever the user passes to
    ctx.checkpoint(); it is returned by ctx.checkpoint_state on the next
    attempt. Carry serializable data across worker boundaries. Recreate
    process-local clients and file handles inside process().
    """

    def __init__(self, *, name: str | None = None) -> None:
        if type(self).process is Stage.process:
            raise TypeError(f"{type(self).__name__} must override process()")
        self._name = type(self).__name__ if name is None else name
        _name(self._name, "stage name")

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def config_dependencies(cls, cfg: Mapping[str, Any]) -> Mapping | bool:
        """Declare which configuration positions this Stage result depends on.

        The declaration mirrors the configuration tree:

        - True at a node depends on the whole subtree at that position;
        - a mapping recurses into its keys (strings for mappings, integers for
          sequence indices);
        - False or an absent key declares no dependency.

        cfg is the read-only configuration of the current attempt, so a
        branch may declare different dependencies per run. Returning True
        declares the entire configuration. Include the identity of external
        inputs in the configuration when they affect the result.
        """
        return True

    def process(self, data: InputT, ctx: RunContext) -> OutputT:
        """Execute the Stage once and return the next pipeline value.

        The last Stage in a Pipeline must be a synchronous Python method with readable source.
        Let it finish naturally, use a bare return, or write return None.
        Record scalar results with ctx.log_metrics() or save artifacts in user code.

        Use ctx.resume_range(start, stop) to obtain a resumable iterator and
        ctx.checkpoint(position, state) to persist state explicitly. The user
        controls the loop, the saved state shape and the resume positions.
        """
        raise NotImplementedError("Stage subclasses must override process()")

    @final
    def run(self, data: InputT, ctx: RunContext | None = None) -> OutputT:
        """Observe a direct invocation. Use a managed Pipeline for checkpointing."""
        context = RunContext() if ctx is None else ctx
        with stage_scope(self.name, context) as child:
            return self.process(data, child)
