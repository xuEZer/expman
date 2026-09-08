"""Sequential composition of user-defined stages."""

from collections.abc import Iterable
from typing import Any

from .context import RunContext, _name
from .stage import Stage


class Pipeline:
    """An ordered snapshot of stages, executed synchronously and fail-fast.

    An empty pipeline returns its input unchanged. Adjacent stage input/output
    compatibility is the caller's responsibility, as Python values may be
    arbitrary objects. The tuple freezes composition, not individual stages.
    """

    def __init__(
        self, stages: Iterable[Stage[Any, Any]], *, name: str = "pipeline"
    ) -> None:
        _name(name, "pipeline name")
        self._stages = tuple(stages)
        if any(not isinstance(stage, Stage) for stage in self._stages):
            raise TypeError("every pipeline element must be a Stage instance")
        self._name = name

    @property
    def stages(self) -> tuple[Stage[Any, Any], ...]:
        return self._stages

    @property
    def name(self) -> str:
        return self._name

    def run(self, data: Any = None, ctx: RunContext | None = None) -> Any:
        context = RunContext() if ctx is None else ctx
        with context.observe(self.name, kind="pipeline") as pipeline_context:
            for stage in self.stages:
                data = stage.run(data, pipeline_context)
            return data
