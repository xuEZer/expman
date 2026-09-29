"""The user extension point for pipeline data processing."""

from collections.abc import Mapping
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, final

from .context import RunContext, _name

if TYPE_CHECKING:
    from .storage import Checkpoint


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


class _CfgValue:
    """A loop() max_iter placeholder resolved from ctx.cfg when the Stage runs."""

    __slots__ = ("key",)

    def __init__(self, key: str) -> None:
        _name(key, "configuration key")
        self.key = key


def _overrides(cls: type, name: str) -> bool:
    """Whether a class below Stage in the MRO defines ``name`` itself."""
    return any(name in vars(base) for base in cls.__mro__ if base is not Stage)


class Stage[InputT, OutputT]:
    """Define exactly one of process() or loop(); run() is the execution entry.

    With loop(), the framework builds the execution around it: init() declares
    the state variables, every iteration is checkpointed automatically, the
    newest checkpoint is the resume point, and the last iteration's return
    value is the Stage output. With process(), the user owns the whole
    execution and the Stage behaves like an ordinary transformation; defining
    both raises TypeError at class definition.

    Stage instances may hold state. Pipeline constructs them with no arguments
    on each execution; read the current experiment configuration from ctx.cfg.
    Calling an instance directly through run preserves that instance's state.

    State variables are the attributes created in init(); every value created
    there is captured into each checkpoint and the completed snapshot. A
    resume skips init() entirely: the checkpoint provides their values.
    Process-bound resources -- dataloaders, clients, file handles -- do not
    belong in init(): the upstream Stage packs them into its output, and
    loop() receives them as data.
    """

    _loop_max_iter: int | _CfgValue | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        defines_process = _overrides(cls, "process")
        defines_loop = _overrides(cls, "loop")
        if defines_process and defines_loop:
            raise TypeError(
                f"{cls.__qualname__} cannot define both process() and loop(); "
                "the framework builds process() around loop()"
            )
        if defines_loop:
            parameter = signature(cls.loop).parameters.get("max_iter")
            if parameter is None or parameter.default is Parameter.empty:
                raise TypeError(
                    f"{cls.__qualname__}.loop must declare max_iter with a "
                    "default value: a nonnegative integer or Stage.cfg(key)"
                )
            default = parameter.default
            if not isinstance(default, _CfgValue):
                _nonnegative_integer(default, f"{cls.__qualname__}.loop max_iter")
            cls._loop_max_iter = default

    def __init__(self, *, name: str | None = None) -> None:
        if not (_overrides(type(self), "process") or _overrides(type(self), "loop")):
            raise TypeError(
                f"{type(self).__name__} must define either process() or loop()"
            )
        self._name = type(self).__name__ if name is None else name
        _name(self._name, "stage name")
        self._state_keys: tuple[str, ...] = ()
        self._loop_step = 0
        self._loop_result: Any = None
        self._break_requested = False
        self._checkpointer: Checkpoint | None = None

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def cfg(cls, key: str) -> _CfgValue:
        """Declare loop()'s max_iter as ctx.cfg[key], resolved when the Stage runs."""
        return _CfgValue(key)

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

    def init(self, ctx: RunContext) -> None:  # noqa: B027 - optional hook
        """Create the state variables; runs when the attempt starts from scratch.

        Attributes created in init() are the state variables: the framework
        snapshots them into every checkpoint and the completed Stage record.
        On a resume init() is skipped entirely and the checkpoint provides
        their values. Attributes whose names start with an underscore, and
        anything that existed before init() ran, are framework bookkeeping and
        never captured.
        """

    def loop(  # noqa: B027 - the hook loop Stages override
        self, data: Any, ctx: RunContext, index: int, max_iter: int | None = None
    ) -> None:
        """Run one iteration; override in a loop Stage, never call it yourself.

        ``data`` is the upstream object handed to the Stage, ``index`` is the
        flat, monotonic iteration position across resumes, and ``max_iter`` is
        the total iteration count declared in the signature -- a plain integer
        or ``Stage.cfg(key)`` resolved from ctx.cfg when the Stage runs; the
        framework passes the resolved value on every call. After the call
        returns normally the framework saves a checkpoint with step=index+1
        holding the state attributes, the random state and the returned value,
        so raising inside loop() discards only the current iteration. The last
        return value becomes the Stage output.
        """
        raise NotImplementedError("loop Stages must override loop()")

    def break_loop(self) -> None:
        """Stop the loop before its next iteration.

        The flag is deliberately not persisted: after a resume, loop() logic
        re-evaluates whether to stop again.
        """
        self._break_requested = True

    def process(self, data: InputT, ctx: RunContext) -> Any:
        """Framework-built execution for loop Stages; override process() to replace it.

        A scratch attempt runs init() and drives loop() from step zero; a
        resume skips init(), restores the checkpoint's state attributes and
        random state, and continues from its step. Iteration positions below
        the recorded step are silently skipped, max_iter is the total number
        of iterations across resumes, and a break_loop() request stops before
        the next iteration.
        """
        total = self._resolve_max_iter(ctx)
        if self._loop_step >= total:
            return self._loop_result
        result = None
        while self._loop_step < total:
            if self._break_requested:
                break
            result = self.loop(data, ctx, self._loop_step, max_iter=total)
            if self._checkpointer is not None:
                self._checkpointer.save(
                    step=self._loop_step + 1, state=self._state(), result=result
                )
            self._loop_step += 1
        return result

    def _resolve_max_iter(self, ctx: RunContext) -> int:
        default = type(self)._loop_max_iter
        if isinstance(default, _CfgValue):
            try:
                value = ctx.cfg[default.key]
            except KeyError:
                raise KeyError(
                    f"{self.name}.loop requires cfg[{default.key!r}], which the "
                    "current configuration does not define"
                ) from None
        else:
            value = default
        return _nonnegative_integer(value, f"{self.name}.loop max_iter")

    def _state(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._state_keys}

    def _prepare(self, ctx: RunContext) -> None:
        self._checkpointer = ctx._checkpoint
        self._break_requested = False
        record = ctx._checkpoint_record
        if record is not None and type(self).process is Stage.process:
            # Loop-mode resume: init() is skipped entirely and the checkpoint
            # provides the initial values of every state variable.
            for key, value in record["state"].items():
                setattr(self, key, value)
            self._state_keys = tuple(record["state"])
            self._loop_step = record["step"] or 0
            self._loop_result = record.get("loop_result")
            if self._checkpointer is not None:
                self._checkpointer.restore_random(record)
            return
        self._loop_result = None
        before = set(self.__dict__)
        self.init(ctx)
        self._state_keys = tuple(
            key
            for key in self.__dict__
            if key not in before and not key.startswith("_")
        )
        self._loop_step = 0

    @final
    def run(self, data: InputT, ctx: RunContext | None = None) -> OutputT:
        """Execute init and process once, with stage-level timing and outcome events."""
        context = RunContext() if ctx is None else ctx
        with context.observe(self.name, kind="stage") as stage_context:
            self._prepare(stage_context)
            output = self.process(data, stage_context)
            if context._store is not None:
                checkpointer = stage_context._checkpoint
                context._store.complete(
                    context._stage_path,
                    output,
                    self._state(),
                    self.name,
                    dependencies=None
                    if checkpointer is None
                    else checkpointer.dependencies,
                )
            return output
