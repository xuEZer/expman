"""Command-line interface: run, resume, inspect and stop experiment Batches.

The pipeline is located as ``module:attr`` on the command line and recorded
in the Batch manifest, so ``expman resume`` can rebuild it without the user
repeating it. Every batch directory under ``runs/`` gets a small integer ID
recorded in ``runs/.batches.sqlite3``: ``expman status`` without arguments keeps
redrawing a live overview of all of them until Ctrl+C, and the same ID is a
shortcut for ``expman stop`` and ``expman resume``. ``expman web`` serves a
dashboard over the same IDs from its own process, independent of any batch
it watches. Retry counts and interval coverage are process-wide settings
configured in code with ``expman.set(...)`` next to the Pipeline, not
command-line flags.
"""

import argparse
import importlib
import sys
from pathlib import Path
from time import sleep

from expman.core.events import Status
from expman.core.pipeline import Pipeline
from expman.estimation.estimate import _clock as _fmt_elapsed
from expman.observation.registry import RUNS_DIR, ensure_ids
from expman.observation.status import RecordedBatch
from expman.process_control import request_stop
from expman.scheduling.batch import Batch
from expman.storage.base import StorageError
from expman.storage.batch import BatchStore
from expman.webui.server import DEFAULT_PORT, serve

_STATUS_INTERVAL = 2.0


def _resolve_pipeline(spec: str) -> Pipeline:
    """Import the Pipeline object named by ``module:attr``."""
    module_name, separator, attr_name = spec.rpartition(":")
    if not module_name or not separator or not attr_name:
        raise SystemExit(f"pipeline must look like 'module:attr', got {spec!r}")
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
    return attribute


def _default_output_dir(spec: str, config: str) -> Path:
    """runs/<pipeline-attr>_<config-stem>, suffixed _2, _3... on collisions."""
    attr = spec.rpartition(":")[2] or "batch"
    stem = Path(config).stem or "cfg"
    base = f"{attr}_{stem}"
    candidate = RUNS_DIR / base
    suffix = 2
    while candidate.exists():
        candidate = RUNS_DIR / f"{base}_{suffix}"
        suffix += 1
    return candidate


def _resolve_directory(token: str) -> Path:
    """Interpret the argument as a batch ID (``0``, ``1``, ...) or a path."""
    if token.isdigit():
        ids = ensure_ids()
        directory = ids.get(int(token))
        if directory is None:
            known = ", ".join(str(i) for i in sorted(ids)) or "none"
            raise SystemExit(
                f"unknown batch ID {token} (known: {known}); see expman status"
            )
        return directory
    return Path(token)


def _timing(report: RecordedBatch) -> tuple[str, str]:
    elapsed = _fmt_elapsed(report.elapsed_seconds, upper=False)
    remaining = str(report.remaining) if report.remaining is not None else "??:??:??"
    return elapsed, remaining


def _execute(batch: Batch, *, progress: bool, resume_command: str) -> int:
    """Run the batch; a stop signal maps to a resume hint and exit 143."""
    print(f"Batch directory: {batch.output_dir}", flush=True)

    try:
        results = batch.run(progress=progress, handle_sigterm=True)
    except KeyboardInterrupt:
        print(f"Stopped; resume with: {resume_command}", file=sys.stderr)
        return 143
    except StorageError as error:
        print(error, file=sys.stderr)
        return 1
    for result in results:
        print(f"{result.run_id}: {result.status.value}")

    return int(any(result.status is Status.FAILED for result in results))


def _cmd_run(args: argparse.Namespace) -> int:
    pipeline = _resolve_pipeline(args.pipeline)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else _default_output_dir(args.pipeline, args.config)
    )
    batch = Batch(
        pipeline,
        args.config,
        output_dir=output_dir,
        _pipeline_spec=args.pipeline,
    )
    ids = ensure_ids()
    batch_id = next(
        (i for i, directory in ids.items() if directory == batch.output_dir), None
    )
    display = str(batch_id) if batch_id is not None else str(batch.output_dir)
    return _execute(
        batch,
        progress=args.progress,
        resume_command=f"expman resume {display}",
    )


def _cmd_resume(args: argparse.Namespace) -> int:
    directory = _resolve_directory(args.directory)
    try:
        manifest = BatchStore(directory).read_manifest()
    except StorageError as error:
        raise SystemExit(str(error)) from None
    spec = manifest.get("pipeline_spec")
    if not isinstance(spec, str):
        raise SystemExit(
            f"{directory} records no pipeline location; resume it from Python "
            "with Batch.resume(pipeline, ...)"
        )
    pipeline = _resolve_pipeline(spec)
    batch = Batch.resume(pipeline, directory)
    return _execute(
        batch,
        progress=args.progress,
        resume_command=f"expman resume {args.directory}",
    )


def _cmd_status(args: argparse.Namespace) -> int:
    if args.directory is None:
        return _live_overview()
    directory = _resolve_directory(args.directory)
    try:
        report = RecordedBatch.read(directory)
    except StorageError as error:
        raise SystemExit(str(error)) from None
    total = report.stage_count
    print(f"Batch {directory} | stages: {total} | runs: {len(report.runs)}")
    for run in report.runs:
        stage = run.stage + 1 if run.stage is not None and total else "-"
        history = " ".join(str(item["status"]).lower() for item in run.attempts)
        print(
            f"  {run.run_id}  {run.state:<10} stage {stage}/{total}"
            f"  attempts: {history or 'none'}"
        )
    elapsed, remaining = _timing(report)
    print(f"elapsed: {elapsed} | remaining: {remaining}")
    if args.directory.isdigit():
        print(f"detail page: `expman web`, then /b/{args.directory}")
    return 0


def _overview_line(batch_id: int, directory: Path) -> str:
    """One summary line of the runs/ overview, tolerating live rewriting."""
    name = str(RUNS_DIR / directory.name)
    try:
        report = RecordedBatch.read(directory)
    except StorageError:
        return f"{batch_id:>2}  {name:<30} unreadable records"
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


def _print_overview() -> None:
    ids = ensure_ids()
    print(
        f"batches under {RUNS_DIR}/"
        f" (refreshing every {_STATUS_INTERVAL:g}s; Ctrl+C to exit)"
    )
    if not ids:
        print("  no batches yet")
        return
    print(
        f"{'ID':>2}  {'BATCH':<30} {'RUNS':>4} {'RUN':>4} {'PEND':>4}"
        f" {'DONE':>4} {'FAIL':>4} {'STOP':>4}  {'STAGE':<8}{'ELAPSED':<10}REMAINING"
    )
    for batch_id, directory in sorted(ids.items()):
        print(_overview_line(batch_id, directory))


def _live_overview() -> int:
    """Keep redrawing the runs/ overview until the user presses Ctrl+C."""
    try:
        while True:
            print("\033[2J\033[H", end="", flush=True)
            _print_overview()
            sleep(_STATUS_INTERVAL)
    except KeyboardInterrupt:
        pass
    return 0


def _cmd_stop(args: argparse.Namespace) -> int:
    directory = _resolve_directory(args.directory)
    success, message = request_stop(directory)
    print(message, file=sys.stderr if not success else None)
    return 0 if success else 1


def _cmd_web(args: argparse.Namespace) -> int:
    """Serve the run dashboard until Ctrl+C; it never runs an experiment.

    The dashboard is its own process on purpose: it outlives every batch it
    watches, and batches started later show up without restarting it.
    """
    root = None if args.root is None else Path(args.root)
    try:
        dashboard = serve(
            root, host=args.host, port=args.port, allow_stop=not args.read_only
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"could not start the dashboard: {error}") from None
    print(f"Dashboard: {dashboard.url}", flush=True)
    print(f"Watching batches under {root or RUNS_DIR}; Ctrl+C to stop.", flush=True)
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
    run.add_argument("pipeline", help="pipeline location as 'module:attr'")
    run.add_argument("config", help="experiment YAML path")
    run.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="batch directory (default: runs/<pipeline>_<config>)",
    )
    resume = commands.add_parser("resume", help="continue a stopped Batch")
    resume.add_argument("directory", help="batch ID from expman status, or a path")
    for command in (run, resume):
        command.add_argument(
            "--progress",
            action="store_true",
            help="show live progress while running (off by default)",
        )
    run.set_defaults(handler=_cmd_run)
    resume.set_defaults(handler=_cmd_resume)

    web = commands.add_parser(
        "web",
        help="serve the dashboard for every batch under runs/",
    )
    web.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="dashboard port (default: 8765; 0 picks a free one)",
    )
    web.add_argument(
        "--host",
        default="127.0.0.1",
        help="dashboard bind address (default: loopback only)",
    )
    web.add_argument(
        "--root",
        default=None,
        help="directory holding the batches (default: runs/)",
    )
    web.add_argument(
        "--read-only",
        action="store_true",
        help="refuse stop requests and hide the stop button",
    )
    web.set_defaults(handler=_cmd_web)

    status = commands.add_parser(
        "status",
        help="show progress; without an ID, watch every batch under runs/",
    )
    status.add_argument(
        "directory",
        nargs="?",
        default=None,
        help="batch ID or path; omit to watch runs/ until Ctrl+C",
    )
    status.set_defaults(handler=_cmd_status)

    stop = commands.add_parser("stop", help="gracefully stop a running Batch")
    stop.add_argument("directory", help="batch ID from expman status, or a path")
    stop.set_defaults(handler=_cmd_stop)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Console entry point; the return value is the process exit status."""
    args = _parser().parse_args(argv)
    return args.handler(args)
