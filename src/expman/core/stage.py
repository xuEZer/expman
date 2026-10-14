"""The user extension point for pipeline data processing."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, final

from expman.core.context import RunContext, _name

if TYPE_CHECKING:
    from expman.storage.base import Checkpoint


class Stage[InputT, OutputT]:
    """Override process(); run() is the execution entry.

    A Stage is a single data transformation. The user owns the whole execution:
    loop control, checkpoint timing and state shape are explicit through
    ctx.resume_range() and ctx.checkpoint(). Pipeline constructs Stage
    instances with no arguments on each execution; read the current experiment
    configuration from ctx.cfg.

    State carried through a resume is whatever the user passes to
    ctx.checkpoint(); it is returned by ctx.checkpoint_state on the next
    attempt. Process-bound resources -- dataloaders, clients, file handles -- do
    not belong in instance attributes that are expected to survive a process
    restart: the upstream Stage packs them into its output, or the user rebuilds
    them inside process() from explicit checkpoint state.
    """

    def __init__(self, *, name: str | None = None) -> None:
        if not self._defines_process():
            raise TypeError(
                f"{type(self).__name__} must override process(); "
                "loop() is no longer supported"
            )
        self._name = type(self).__name__ if name is None else name
        _name(self._name, "stage name")
        self._checkpointer: Checkpoint | None = None

    def _defines_process(self) -> bool:
        return type(self).process is not Stage.process

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
        declares the entire configuration. The default is conservative: a Stage
        that reads ctx.cfg without overriding this method never reuses stale
        results.
        """
        return True

    def process(self, data: InputT, ctx: RunContext) -> OutputT:
        """Execute the Stage once and return the next pipeline value.

        Use ctx.resume_range(start, stop) to obtain a resumable iterator and
        ctx.checkpoint(position, state) to persist state explicitly. The user
        controls the loop, the saved state shape and the resume positions.
        """
        raise NotImplementedError("Stage subclasses must override process()")

    def _prepare(self, ctx: RunContext) -> None:
        self._checkpointer = ctx._checkpoint
        record = ctx._checkpoint_record
        if record is not None and self._checkpointer is not None:
            # On a resume the random stream is rewound to the checkpoint moment;
            # business state reaches the process() through ctx.checkpoint_state.
            self._checkpointer.restore_random(record)

    @final
    def run(self, data: InputT, ctx: RunContext | None = None) -> OutputT:
        """Execute process once, with stage-level timing and outcome events."""
        context = RunContext() if ctx is None else ctx
        with context.observe(self.name, kind="stage") as stage_context:
            self._prepare(stage_context)
            output = self.process(data, stage_context)
            if context._store is not None:
                checkpointer = stage_context._checkpoint
                if stage_context._explicit_checkpoint_used:
                    user_state = stage_context.checkpoint_state
                    state = (
                        user_state
                        if isinstance(user_state, dict)
                        else {"__payload__": user_state}
                    )
                else:
                    state = {}
                context._store.complete(
                    context._stage_path,
                    output,
                    state,
                    self.name,
                    dependencies=None
                    if checkpointer is None
                    else checkpointer.dependencies,
                )
            return output
