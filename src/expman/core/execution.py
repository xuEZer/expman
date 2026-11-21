"""Stage invocation and durable execution of a Pipeline's stages."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from time import perf_counter
from typing import TYPE_CHECKING, Any

from expman.core.context import RunContext
from expman.core.dependencies import stage_dependencies
from expman.core.events import Status
from expman.storage.base import RecoveryWarning, StorageError
from expman.storage.run import Checkpoint, RunStore

if TYPE_CHECKING:
    from expman.core.stage import Stage


@contextmanager
def stage_scope(
    name: str,
    ctx: RunContext,
    checkpoint: Checkpoint | None = None,
) -> Iterator[RunContext]:
    """Observe processing and output publication as one Stage execution."""
    scoped = replace(ctx, _checkpoint=checkpoint)
    with scoped.observe(name, kind="stage") as child:
        if checkpoint is not None:
            checkpoint.restore_random()
        yield child


class ManagedStages:
    """Restore or execute ordered stages for one durable Pipeline invocation."""

    def __init__(self, store: RunStore, base: tuple[int, ...], signature: list):
        store.bind_pipeline(base, signature)
        self.store = store

    def _restore(self, ctx, snapshot, started, *, shared_reference=None):
        position = ctx._stage_path
        if not snapshot.get("leaf"):
            self.store.restore_random(snapshot)
        if shared_reference is not None:
            self.store.reuse(position, ctx.attempt, shared_reference, snapshot)
        duration = perf_counter() - started
        with ctx.observe(snapshot["name"], kind="stage", reused=True):
            self.store.status(
                position,
                Status.SUCCEEDED.value,
                ctx.attempt,
                reused=True,
                elapsed_seconds=snapshot["elapsed_seconds"],
                restore_seconds=duration,
            )
            return snapshot.get("output")

    def run_scheduled(self, stages, index: int, ctx: RunContext) -> None:
        """Start the scheduled Stage from its immediate predecessor's output and RNG."""
        position = (index,)
        data = None
        cache_parent = "root" if index == 0 and self.store.shared is not None else None
        if index and not self.store.has_completed(position):
            previous = self.store.completed((index - 1,))
            if previous is None:
                raise StorageError(
                    f"scheduled Stage {index} has no completed predecessor"
                )
            self.store.restore_random(previous)
            data = previous["output"]
            if reference := previous.get("cache_ref"):
                cache_parent = reference[2]
        scoped = replace(ctx, stage_id=index, _stage_path=position)
        self.run(
            stages[index],
            data,
            scoped,
            cache_parent=cache_parent,
            leaf=index == len(stages) - 1,
        )

    def run(
        self,
        stage_type: type[Stage[Any, Any]],
        data: Any,
        ctx: RunContext,
        *,
        cache_parent: str | None = None,
        leaf=False,
    ) -> Any:
        store = self.store
        position = ctx._stage_path
        started = perf_counter()
        snapshot = store.completed(position)
        if snapshot is not None:
            return self._restore(ctx, snapshot, started)

        try:
            dependencies = stage_dependencies(stage_type, ctx.cfg)
            if cache_parent is not None and store.can_reuse(position):
                assert store.shared is not None
                candidate = store.shared.find(
                    cache_parent, position, dependencies, leaf=leaf
                )
                if candidate is not None:
                    reference, snapshot = candidate
                    return self._restore(
                        ctx, snapshot, started, shared_reference=reference
                    )
            store.status(position, Status.RUNNING.value, ctx.attempt)
            checkpoint = Checkpoint(
                store,
                position,
                ctx._pipeline_calls,
                dependencies,
                record=store.latest(position, dependencies=dependencies),
            )
            stage = stage_type()
            with stage_scope(stage.name, ctx, checkpoint) as stage_context:
                output = stage.process(data, stage_context)
                elapsed = store.complete(
                    checkpoint, stage.name, output, cache_parent, leaf=leaf
                )
            store.status(
                position, Status.SUCCEEDED.value, ctx.attempt, elapsed_seconds=elapsed
            )
            return output
        except BaseException as error:
            status = Status.FAILED if isinstance(error, Exception) else Status.CANCELLED
            try:
                store.status(position, status.value, ctx.attempt)
            except Exception as storage_error:
                warnings.warn(
                    f"cannot record stage status: {storage_error}",
                    RecoveryWarning,
                    stacklevel=2,
                )
            raise
