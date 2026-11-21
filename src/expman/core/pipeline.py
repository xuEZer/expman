"""Sequential composition of user-defined stages."""

import ast
import inspect
import sys
from collections.abc import Iterable
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import CodeType
from typing import Any

from expman.core.context import RunContext, _name
from expman.core.execution import ManagedStages
from expman.core.stage import Stage


def _check_last_stage(stage: type[Stage[Any, Any]]) -> None:
    name = f"{stage.__qualname__}.process"
    code = getattr(stage.process, "__code__", None)
    if not isinstance(code, CodeType):
        raise ValueError(f"{name} must be a Python method with readable source")
    location = f"{code.co_filename}:{code.co_firstlineno}"
    if code.co_flags & (
        inspect.CO_COROUTINE | inspect.CO_GENERATOR | inspect.CO_ASYNC_GENERATOR
    ):
        raise ValueError(f"{location}: {name} must be synchronous and must return None")

    # Passing the code object checks the actual implementation of a decorated method.
    try:
        lines, first_line = inspect.getsourcelines(code)
        indent = lines[0][: len(lines[0]) - len(lines[0].lstrip())]
        body = ast.parse("".join(line.removeprefix(indent) for line in lines)).body
    except (OSError, TypeError, SyntaxError) as error:
        raise ValueError(
            f"{location}: cannot read the Python definition of {name}"
        ) from error
    if (
        len(body) != 1
        or not isinstance(body[0], ast.FunctionDef)
        or body[0].name != code.co_name
    ):
        raise ValueError(f"{location}: define {name} with a synchronous def")

    pending: list[ast.AST] = list(reversed(body[0].body))
    while pending:
        statement = pending.pop()
        if isinstance(statement, ast.Return):
            value = statement.value
            if value is not None and not (
                isinstance(value, ast.Constant) and value.value is None
            ):
                line = first_line + statement.lineno - 1
                raise ValueError(
                    f"{code.co_filename}:{line}: last stage {name} must return None"
                )
        elif not isinstance(
            statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            # Returns in nested functions and classes belong to those definitions.
            pending.extend(reversed(list(ast.iter_child_nodes(statement))))


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

    The last Stage must return None. An empty pipeline returns its input unchanged.
    Adjacent stages can pass arbitrary Python objects. The caller is responsible
    for matching their input and output types. Each Stage is constructed without
    arguments when it needs to run.
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
        if self._stages:
            _check_last_stage(self._stages[-1])
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
        """Run the full Pipeline, or return the selected Stage's output at stop_after."""
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
