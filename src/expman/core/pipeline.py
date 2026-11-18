"""Sequential composition of user-defined stages."""

import sys
from collections.abc import Iterable
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

from expman.core.context import RunContext, _name
from expman.core.execution import ManagedStages
from expman.core.stage import Stage


def _module_source_hash(stage: type[Stage[Any, Any]]) -> str | None:
    """Fingerprint the containing module in both the scheduler and fresh workers.

    Editing the file changes the signature for every Stage defined in it.
    Dynamic modules without a readable file contribute only the class identity.
    """
    module = sys.modules.get(stage.__module__)
    path = getattr(module, "__file__", None)
    if not path:
        return None
    try:
        return sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


class Pipeline:
    """A reusable sequence of Stage classes, executed synchronously and fail-fast.

    An empty pipeline returns its input unchanged. Adjacent stage input/output
    compatibility is the caller's responsibility, as Python values may be
    arbitrary objects. Each Stage that needs to compute an output is
    constructed without arguments. A restored output skips construction.
    """

    def __init__(
        self, stages: Iterable[type[Stage[Any, Any]]], *, name: str = "pipeline"
    ) -> None:
        _name(name, "pipeline name")
        self._stages = tuple(stages)
        if any(
            not isinstance(stage, type) or not issubclass(stage, Stage)
            for stage in self._stages
        ):
            raise TypeError("every pipeline element must be a Stage class")
        self._name = name

    @property
    def stages(self) -> tuple[type[Stage[Any, Any]], ...]:
        return self._stages

    @property
    def name(self) -> str:
        return self._name

    def signature(self) -> list[dict]:
        result = []
        for stage in self.stages:
            result.append(
                {
                    "class": f"{stage.__module__}.{stage.__qualname__}",
                    "source": _module_source_hash(stage),
                }
            )
        return result

    def run(
        self,
        data: Any = None,
        ctx: RunContext | None = None,
        *,
        stop_after: int | None = None,
    ) -> Any:
        """Run through ``stop_after`` when supplied, otherwise the full Pipeline."""
        if stop_after is not None and not 0 <= stop_after < len(self.stages):
            raise ValueError("stop_after must identify a Pipeline stage")
        context = RunContext() if ctx is None else ctx
        base = context._stage_path
        if context.stage_id is not None:
            invocation = context._pipeline_calls.get(base, 0)
            context._pipeline_calls[base] = invocation + 1
            base = (*base, invocation)
        managed = (
            ManagedStages(context._store, base, self.signature())
            if context._store is not None
            else None
        )
        stages = self.stages if stop_after is None else self.stages[: stop_after + 1]
        with context.observe(self.name, kind="pipeline") as pipeline_context:
            for index, stage in enumerate(stages):
                scoped = replace(
                    pipeline_context, stage_id=index, _stage_path=(*base, index)
                )
                if managed is None:
                    data = stage().run(data, scoped)
                else:
                    data = managed.run(stage, data, scoped)
        return data
