"""Sequential composition of user-defined stages."""

import warnings
from collections.abc import Iterable
from dataclasses import replace
from hashlib import sha256
from inspect import getsource
from typing import Any

from .context import RunContext, _name
from .events import Status
from .stage import Stage
from .storage import Checkpoint, RecoveryWarning


class Pipeline:
    """A reusable sequence of Stage classes, executed synchronously and fail-fast.

    An empty pipeline returns its input unchanged. Adjacent stage input/output
    compatibility is the caller's responsibility, as Python values may be
    arbitrary objects. Each execution constructs fresh, zero-argument stages.
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
            try:
                source = sha256(getsource(stage).encode()).hexdigest()
            except (OSError, TypeError):
                source = None
            result.append(
                {"class": f"{stage.__module__}.{stage.__qualname__}", "source": source}
            )
        return result

    def run(self, data: Any = None, ctx: RunContext | None = None) -> Any:
        context = RunContext() if ctx is None else ctx
        base = context._stage_path
        if context.stage_id is not None:
            invocation = context._pipeline_calls.get(base, 0)
            context._pipeline_calls[base] = invocation + 1
            base = (*base, invocation)
        if context._store is not None:
            context._store.bind_pipeline(base, self.signature())
        with context.observe(self.name, kind="pipeline") as pipeline_context:
            for index, stage_type in enumerate(self.stages):
                position = (*base, index)
                scoped = replace(
                    pipeline_context,
                    stage_id=index,
                    _stage_path=position,
                    _checkpoint=None,
                )
                store = scoped._store
                if store is not None:
                    completed = store.completed(position)
                    if completed is not None:
                        with scoped.observe(
                            completed["name"], kind="stage", reused=True
                        ):
                            scoped.state.clear()
                            scoped.state.update(completed["state"])
                            data = completed["output"]
                            store.status(
                                position,
                                Status.SUCCEEDED.value,
                                context.attempt,
                                reused=True,
                            )
                        continue
                try:
                    if store is not None:
                        store.status(position, Status.RUNNING.value, context.attempt)
                        checkpoint = store.latest(position)
                        if checkpoint is not None:
                            scoped.state.clear()
                            scoped.state.update(checkpoint["state"])
                            scoped._pipeline_calls.update(
                                checkpoint.get("pipeline_calls", {})
                            )
                        scoped = replace(
                            scoped,
                            _checkpoint=Checkpoint(
                                store,
                                position,
                                scoped.state,
                                None if checkpoint is None else checkpoint.get("step"),
                                scoped._pipeline_calls,
                            ),
                        )
                    data = stage_type().run(data, scoped)
                    if store is not None:
                        store.status(position, Status.SUCCEEDED.value, context.attempt)
                except BaseException as error:
                    if store is not None:
                        status = (
                            Status.FAILED
                            if isinstance(error, Exception)
                            else Status.CANCELLED
                        )
                        try:
                            store.status(position, status.value, context.attempt)
                        except Exception as storage_error:
                            warnings.warn(
                                f"cannot record stage status: {storage_error}",
                                RecoveryWarning,
                                stacklevel=2,
                            )
                    raise
            return data
