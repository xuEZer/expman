"""Command-line interface: run, resume, inspect, stop and delete experiment Batches.

The CLI resolves a bare pipeline name in the configured ``pipeline_module``
(built-in default ``pipeline``) or accepts an explicit ``module:attr``.
The run command reads ``./configs/<config>.yaml`` for the given config name.
The batch manifest records the resolved ``module:attr`` for
``expman resume``. Each batches directory (default ``./batches/``) has its own
integer IDs recorded in ``.batches.sqlite3``. ``expman status`` refreshes the
directory overview or the selected batch's details until Ctrl+C. An ID also
selects a batch for ``stop``, ``resume`` or ``delete``. ``expman web`` serves a
dashboard over the same IDs from its own process, independent of any batch
it watches. The CLI reads ``[tool.expman]`` from the working directory's
``pyproject.toml``. The pipeline module can override settings when imported.
"""

import argparse
import importlib
import shlex
import sys
from functools import partial
from pathlib import Path
from time import sleep, strftime

from expman.config.loader import _load_configs
from expman.config.schema import CONFIGS_DIR, _safe_segment
from expman.core.events import Status
from expman.core.pipeline import Pipeline
from expman.estimation.estimate import _clock as _fmt_elapsed
from expman.observation.registry import delete_batch, ensure_ids
from expman.observation.status import RecordedBatch
from expman.process_control import request_stop
from expman.scheduling.batch import _Batch
from expman.settings import Defaults, project_settings, settings
from expman.storage.batch import BatchStore
from expman.webui.server import serve


def _resolve_pipeline(spec: str, default_module: str) -> tuple[str, Pipeline]:
    """Import the pipeline and return its fully qualified location."""
    module_name, separator, attr_name = spec.rpartition(":")
    if not separator:
        module_name = default_module
    if not module_name or not attr_name or Path(attr_name).name != attr_name:
        raise SystemExit(
            f"pipeline must look like 'attr' or 'module:attr', got {spec!r}"
        )
    if "" not in sys.path:
        sys.path.insert(0, "")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise SystemExit(
            f"could not import pipeline module {module_name!r}: {error}"
        ) from None
    try:
        attribute: object = getattr(module, attr_name)
    except AttributeError:
        raise SystemExit(
            f"module {module_name!r} defines no attribute {attr_name!r}"
        ) from None
    if not isinstance(attribute, Pipeline):
        raise SystemExit(f"{spec!r} must be a Pipeline; got {type(attribute).__name__}")
    return f"{module_name}:{attr_name}", attribute


def _experiment_yaml(name: str) -> Path:
    if not _safe_segment(name) or name.lower().endswith((".yaml", ".yml")):
        raise argparse.ArgumentTypeError(
            "expected a config name without a directory or YAML extension, e.g. baseline"
        )
    directory = CONFIGS_DIR.resolve()
    path = directory / f"{name}.yaml"
    if not path.resolve().is_relative_to(directory):
        raise argparse.ArgumentTypeError(f"{name!r} points outside ./configs/")
    return path


def _command_hint(
    command: str, batches_dir: Path, *arguments: str, default_dir: Path | None = None
) -> str:
    words = ["expman", command, *arguments]
    directory = batches_dir.resolve()
    default = settings.batches_dir if default_dir is None else default_dir
    if directory != Defaults().batches_dir.resolve() or directory != default.resolve():
        words.extend(["--batches-dir", str(directory)])
    return shlex.join(words)


def _batch_directory(batches_dir: Path, batch_id: int) -> Path:
    directory = ensure_ids(batches_dir).get(batch_id)
    if directory is None:
        hint = _command_hint("status", batches_dir)
        raise SystemExit(f"unknown batch ID {batch_id} under {batches_dir}; see {hint}")
    return directory


def _timing(report: RecordedBatch) -> tuple[str, str]:
    elapsed = _fmt_elapsed(report.elapsed_seconds, upper=False)
    remaining = str(report.remaining) if report.remaining is not None else "??:??:??"
    return elapsed, remaining


def _execute(batch: _Batch, *, batch_id: int, default_dir: Path) -> int:
    """Run the batch; a stop signal maps to a resume hint and exit 143."""
    print(f"Batch ID: {batch_id}\nBatch directory: {batch.output_dir}", flush=True)

    try:
        results = batch.run()
    except KeyboardInterrupt:
        resume_command = _command_hint(
            "resume", batch.output_dir.parent, str(batch_id), default_dir=default_dir
        )
        print(f"Stopped; resume with: {resume_command}", file=sys.stderr)
        return 143
    for result in results:
        print(f"{result.run_id}: {result.status.value}")

    return int(any(result.status is Status.FAILED for result in results))


def _cmd_run(args: argparse.Namespace) -> int:
    default_dir = settings.batches_dir.resolve()
    spec, pipeline = _resolve_pipeline(args.pipeline, settings.pipeline_module)
    attr = spec.rpartition(":")[2]
    stem = args.config.stem
    timestamp = strftime("%Y%m%d%H%M%S")
    parent = args.batches_dir if args.batches_dir is not None else settings.batches_dir
    directory = parent / f"{attr}_{stem}_{timestamp}"
    batch = _Batch.create(
        pipeline,
        _load_configs(args.config),
        output_dir=directory,
        pipeline_spec=spec,
    )
    ids = ensure_ids(batch.output_dir.parent)
    batch_id = next(i for i, path in ids.items() if path == batch.output_dir)
    return _execute(batch, batch_id=batch_id, default_dir=default_dir)


def _cmd_resume(args: argparse.Namespace) -> int:
    default_dir = settings.batches_dir.resolve()
    directory = _batch_directory(args.batches_dir, args.ID)
    store = BatchStore(directory)
    manifest = store.read_resumable_manifest()
    # A short name in an existing manifest was recorded against the built-in module.
    _, pipeline = _resolve_pipeline(
        manifest["pipeline_spec"], Defaults().pipeline_module
    )
    batch = _Batch.resume(pipeline, store, manifest)
    return _execute(batch, batch_id=args.ID, default_dir=default_dir)


def _print_detail(directory: Path, batch_id: int) -> None:
    report = RecordedBatch.read(directory)
    total = report.stage_count
    print(f"Batch {batch_id}: {directory} | stages: {total} | runs: {len(report.runs)}")
    for run in report.runs:
        stage = run.stage + 1 if run.stage is not None and total else "-"
        history = " ".join(str(item["status"]).lower() for item in run.attempts)
        print(
            f"  {run.run_id}  {run.state:<10} stage {stage}/{total}"
            f"  attempts: {history or 'none'}"
        )
    elapsed, remaining = _timing(report)
    print(f"elapsed: {elapsed} | remaining: {remaining}")
    print(
        f"detail page: `{_command_hint('web', directory.parent)}`, then /b/{batch_id}"
    )


def _overview_line(batch_id: int, directory: Path) -> str:
    """Format one batch's saved state for the status overview."""
    name = str(directory)
    report = RecordedBatch.read(directory)
    counts = report.counts
    reached = max(
        (run.stage for run in report.runs if run.stage is not None), default=None
    )
    stage = reached + 1 if reached is not None and report.stage_count else "-"
    position = f"{stage}/{report.stage_count}"
    elapsed, remaining = _timing(report)
    return (
        f"{batch_id:>2}  {name:<30} {len(report.runs):>4} "
        f"{counts['running']:>4} {counts['pending']:>4} {counts['succeeded']:>4} "
        f"{counts['failed']:>4} {counts['cancelled']:>4}  "
        f"{position:<8}{elapsed:<10}{remaining}"
    )


def _print_overview(batches_dir: Path) -> None:
    ids = ensure_ids(batches_dir)
    print(f"batches under {batches_dir}")
    if not ids:
        print("  no batches yet")
        return
    print(
        f"{'ID':>2}  {'BATCH':<30} {'RUNS':>4} {'RUN':>4} {'PEND':>4}"
        f" {'DONE':>4} {'FAIL':>4} {'STOP':>4}  {'STAGE':<8}{'ELAPSED':<10}REMAINING"
    )
    for batch_id, directory in sorted(ids.items()):
        print(_overview_line(batch_id, batches_dir / directory.name))


def _cmd_status(args: argparse.Namespace) -> int:
    """Refresh the overview or a single batch's records until Ctrl+C."""
    interval = settings.status_interval
    try:
        if args.ID is None:
            print_frame = partial(_print_overview, args.batches_dir)
        else:
            directory = _batch_directory(args.batches_dir, args.ID)
            print_frame = partial(_print_detail, directory, args.ID)
        while True:
            print("\033[2J\033[H", end="")
            print_frame()
            print(f"Refreshing every {interval:g}s; Ctrl+C to exit.", flush=True)
            sleep(interval)
    except KeyboardInterrupt:
        pass
    return 0


def _cmd_stop(args: argparse.Namespace) -> int:
    directory = _batch_directory(args.batches_dir, args.ID)
    success, message = request_stop(directory)
    print(message, file=sys.stderr if not success else None)
    return 0 if success else 1


def _cmd_delete(args: argparse.Namespace) -> int:
    directory = delete_batch(args.batches_dir, args.ID)
    print(f"Deleted batch {args.ID}: {directory}")
    return 0


def _cmd_web(args: argparse.Namespace) -> int:
    """Serve the run dashboard until Ctrl+C; it never runs an experiment.

    The dashboard is its own process on purpose: it outlives every batch it
    watches, and batches started later show up without restarting it.
    """
    dashboard = serve(args.batches_dir, host=args.host, port=args.port)
    print(f"Dashboard: {dashboard.url}", flush=True)
    print(f"Watching batches under {args.batches_dir}; Ctrl+C to stop.", flush=True)
    try:
        dashboard.wait()
    finally:
        dashboard.close()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="expman", description="Run and manage expman experiment Batches."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="create and run a new Batch")
    run.add_argument(
        "pipeline",
        help=f"pipeline attribute or 'module:attr' (default module: {settings.pipeline_module})",
    )
    run.add_argument(
        "config",
        type=_experiment_yaml,
        metavar="CONFIG",
        help="experiment name, e.g. baseline for ./configs/baseline.yaml",
    )
    resume = commands.add_parser("resume", help="continue a stopped Batch")
    resume.add_argument("ID", type=int, help="batch ID from expman status")
    run.set_defaults(handler=_cmd_run)
    resume.set_defaults(handler=_cmd_resume)

    web = commands.add_parser(
        "web",
        help="serve the dashboard for every batch under the selected root",
    )
    web.add_argument(
        "--port",
        type=int,
        default=settings.web_port,
        help="dashboard port (default: %(default)s; 0 picks a free one)",
    )
    web.add_argument(
        "--host",
        default=settings.web_host,
        help="dashboard bind address (default: %(default)s)",
    )
    web.set_defaults(handler=_cmd_web)

    status = commands.add_parser(
        "status",
        help="watch all batches or one batch until Ctrl+C",
    )
    status.add_argument(
        "ID",
        type=int,
        nargs="?",
        help="watch this batch's runs; omit to watch all batches",
    )
    status.set_defaults(handler=_cmd_status)

    stop = commands.add_parser("stop", help="gracefully stop a running Batch")
    stop.add_argument("ID", type=int, help="batch ID from expman status")
    stop.set_defaults(handler=_cmd_stop)

    delete = commands.add_parser(
        "delete", help="delete an idle batch directory and release its ID"
    )
    delete.add_argument("ID", type=int, help="batch ID from expman status")
    delete.set_defaults(handler=_cmd_delete)

    for command in (run, resume, web, status, stop, delete):
        command.add_argument(
            "--batches-dir",
            type=Path,
            # A run imports the pipeline before choosing its default directory.
            default=None if command is run else settings.batches_dir,
            metavar="DIR",
            help=f"parent directory holding the batches (default: {settings.batches_dir})",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Console entry point; the return value is the process exit status."""
    with project_settings():
        args = _parser().parse_args(argv)
        return args.handler(args)
