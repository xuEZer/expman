"""Command-line interface: run, resume, inspect and stop experiment Batches.

The pipeline is located as ``module:attr`` on the command line and recorded
in the Batch manifest, so ``expman resume`` can rebuild it without the user
repeating it. Every batch directory under ``runs/`` gets a small integer ID
recorded in ``runs/.batches.json``: ``expman status`` without arguments keeps
redrawing a live overview of all of them until Ctrl+C, and the same ID is a
shortcut for ``expman stop`` and ``expman resume``. Retry counts and interval
coverage are process-wide settings configured in code with ``expman.set(...)``
next to the Pipeline, not command-line flags.
"""

import argparse
import importlib
import json
import os
import signal
import sys
from contextlib import suppress
from pathlib import Path
from time import sleep, time

from . import settings
from .batch import Batch
from .estimation import TimeEstimate
from .pipeline import Pipeline
from .stage import Stage
from .storage import PickleSerializer, StorageError, read_record
from .webui import DEFAULT_PORT

_PID_FILE = "expman.pid"
_WEBUI_FILE = ".webui.json"
_RUNS_DIR = Path("runs")
_REGISTRY_NAME = ".batches.json"
_STATUS_INTERVAL = 2.0


def _coerce(attribute: object) -> Pipeline | None:
    """Turn a module attribute into a Pipeline, or None if it is not one."""
    if isinstance(attribute, Pipeline):
        return attribute
    if (
        isinstance(attribute, (list, tuple))
        and bool(attribute)
        and all(
            isinstance(item, type) and issubclass(item, Stage) for item in attribute
        )
    ):
        return Pipeline(list(attribute))
    return None


def _resolve_pipeline(spec: str) -> Pipeline:
    """Import ``module:attr`` and coerce the attribute into a Pipeline."""
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
    pipeline = _coerce(attribute)
    if pipeline is None and callable(attribute) and not isinstance(attribute, type):
        pipeline = _coerce(attribute())
    if pipeline is None:
        raise SystemExit(
            f"{spec!r} must be a Pipeline, a list of Stage classes, or a "
            f"zero-argument factory returning one; got "
            f"{type(attribute).__name__}"
        )
    return pipeline


def _default_output_dir(spec: str, config: str) -> Path:
    """runs/<pipeline-attr>_<config-stem>, suffixed _2, _3... on collisions."""
    attr = spec.rpartition(":")[2] or "batch"
    stem = Path(config).stem or "cfg"
    base = f"{attr}_{stem}"
    candidate = _RUNS_DIR / base
    suffix = 2
    while candidate.exists():
        candidate = _RUNS_DIR / f"{base}_{suffix}"
        suffix += 1
    return candidate


def _load_registry() -> dict:
    """Read the ID registry, tolerating a missing or damaged file."""
    try:
        data = json.loads((_RUNS_DIR / _REGISTRY_NAME).read_text())
    except (OSError, ValueError):
        return {"next_id": 0, "batches": {}}
    if not isinstance(data, dict) or not isinstance(data.get("batches"), dict):
        return {"next_id": 0, "batches": {}}
    if not isinstance(data.get("next_id"), int):
        data["next_id"] = len(data["batches"])
    return data


def _save_registry(registry: dict) -> None:
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (_RUNS_DIR / _REGISTRY_NAME).write_text(json.dumps(registry, indent=2))


def _is_batch_dir(path: Path) -> bool:
    return path.is_dir() and (path / "batch.pkl").exists()


def _scan_runs(registry: dict) -> bool:
    """Register unregistered runs/ batches, oldest first; True if changed."""
    try:
        root = _RUNS_DIR.resolve()
        candidates = sorted(
            (path for path in root.iterdir() if _is_batch_dir(path)),
            key=lambda path: path.stat().st_mtime,
        )
    except OSError:
        return False
    known = set(registry["batches"].values())
    changed = False
    for path in candidates:
        if path.name not in known:
            registry["batches"][str(registry["next_id"])] = path.name
            registry["next_id"] += 1
            changed = True
    return changed


def _ensure_ids() -> dict[int, Path]:
    """Assign stable IDs to runs/ batches; return the ID -> directory map."""
    registry = _load_registry()
    if _scan_runs(registry):
        _save_registry(registry)
    root = _RUNS_DIR.resolve()
    return {
        int(batch_id): root / name
        for batch_id, name in registry["batches"].items()
        if (root / name).is_dir()
    }


def _resolve_directory(token: str) -> Path:
    """Interpret the argument as a batch ID (``0``, ``1``, ...) or a path."""
    if token.isdigit():
        ids = _ensure_ids()
        directory = ids.get(int(token))
        if directory is None:
            known = ", ".join(str(i) for i in sorted(ids)) or "none"
            raise SystemExit(
                f"unknown batch ID {token} (known: {known}); see expman status"
            )
        return directory
    return Path(token)


def _read_manifest(directory: Path) -> dict:
    """Read and sanity-check the persisted Batch manifest."""
    try:
        manifest = read_record(directory / "batch.pkl", PickleSerializer())
    except (StorageError, OSError) as error:
        raise SystemExit(f"cannot read batch records in {directory}: {error}") from None
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("experiments"), list
    ):
        raise SystemExit(f"{directory} does not hold a Batch manifest")
    return manifest


def _fmt_elapsed(seconds: object) -> str:
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        return "??:??:??"
    minutes = int(max(0.0, float(seconds)) // 60)
    days, remainder = divmod(minutes, 24 * 60)
    hours, minutes = divmod(remainder, 60)
    return f"{days:02d}:{hours:02d}:{minutes:02d}"


def _read_timing(directory: Path) -> tuple[str, str]:
    """The persisted elapsed time and remaining-time estimate, if any."""
    elapsed = "??:??:??"
    remaining = "??:??:??"
    timing_path = directory / "timing.pkl"
    if timing_path.exists():
        try:
            timing = read_record(timing_path, PickleSerializer())
        except (StorageError, OSError):
            timing = None
        if isinstance(timing, dict) and timing.get("version") == 1:
            elapsed = _fmt_elapsed(timing.get("elapsed_seconds"))
            saved = timing.get("estimate")
            if isinstance(saved, dict):
                with suppress(TypeError, ValueError):
                    remaining = str(TimeEstimate(**saved))
    return elapsed, remaining


def _start_dashboard(batch: Batch, args: argparse.Namespace) -> str | None:
    """Start the loopback dashboard; a dashboard problem never fails a Batch."""
    if not args.web:
        return None
    if not 0 <= args.web_port <= 65535:
        raise SystemExit("--web-port must be between 0 and 65535")
    from .webui import serve

    try:
        dashboard = serve(
            batch, host=args.web_host, port=args.web_port, allow_stop=True
        )
    except (OSError, ValueError) as error:
        print(f"dashboard not started: {error}", file=sys.stderr)
        return None
    print(f"Dashboard: {dashboard.url}", flush=True)
    try:
        (batch.output_dir / _WEBUI_FILE).write_text(
            json.dumps(
                {"url": dashboard.url, "pid": os.getpid(), "started_at": time()},
                indent=2,
            )
        )
    except OSError as error:
        print(f"could not record the dashboard URL: {error}", file=sys.stderr)
    return dashboard.url


def _forget_dashboard(batch: Batch) -> None:
    with suppress(OSError):
        (batch.output_dir / _WEBUI_FILE).unlink()


def _read_dashboard(directory: Path) -> str | None:
    """The dashboard URL a Batch process recorded, if it wrote one."""
    try:
        data = json.loads((directory / _WEBUI_FILE).read_text())
    except (OSError, ValueError):
        return None
    url = data.get("url") if isinstance(data, dict) else None
    return url if isinstance(url, str) and url else None


def _run_with_dashboard(
    batch: Batch, args: argparse.Namespace, *, progress: bool, resume_command: str
) -> int:
    """Run the Batch, serving its dashboard for as long as the process lives."""
    url = _start_dashboard(batch, args)
    try:
        return _execute(batch, progress=progress, resume_command=resume_command)
    finally:
        if url is not None:
            _forget_dashboard(batch)


def _execute(batch: Batch, *, progress: bool, resume_command: str) -> int:
    """Run the batch; a stop signal maps to a resume hint and exit 143."""
    print(f"Batch directory: {batch.output_dir}", flush=True)

    def stopped(signum, frame):
        raise KeyboardInterrupt()

    # Not the main thread: expman stop then cannot reach this process.
    with suppress(ValueError):
        signal.signal(signal.SIGTERM, stopped)
    try:
        (batch.output_dir / _PID_FILE).write_text(str(os.getpid()))
        results = batch.run(progress=progress)
    except KeyboardInterrupt:
        print(f"Stopped; resume with: {resume_command}", file=sys.stderr)
        return 143
    finally:
        with suppress(FileNotFoundError):
            (batch.output_dir / _PID_FILE).unlink()
    for result in results:
        print(f"{result.run_id}: {result.status.value}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    pipeline = _resolve_pipeline(args.pipeline)
    module_name, _separator, attr_name = args.pipeline.rpartition(":")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else _default_output_dir(args.pipeline, args.config)
    )
    batch = Batch(
        pipeline,
        args.config,
        max_retries=int(settings.get("retries")),
        estimate_coverage=float(settings.get("coverage")),
        output_dir=output_dir,
        _pipeline_spec={"module": module_name, "attr": attr_name},
    )
    ids = _ensure_ids()
    batch_id = next(
        (i for i, directory in ids.items() if directory == batch.output_dir), None
    )
    display = str(batch_id) if batch_id is not None else str(batch.output_dir)
    return _run_with_dashboard(
        batch,
        args,
        progress=args.progress,
        resume_command=f"expman resume {display}",
    )


def _cmd_resume(args: argparse.Namespace) -> int:
    directory = _resolve_directory(args.directory)
    manifest = _read_manifest(directory)
    spec = manifest.get("pipeline_spec")
    if (
        not isinstance(spec, dict)
        or not isinstance(spec.get("module"), str)
        or not (isinstance(spec.get("attr"), str))
    ):
        raise SystemExit(
            f"{directory} records no pipeline location; resume it from Python "
            "with Batch.resume(pipeline, ...)"
        )
    pipeline = _resolve_pipeline(f"{spec['module']}:{spec['attr']}")
    batch = Batch.resume(pipeline, directory)
    return _run_with_dashboard(
        batch,
        args,
        progress=args.progress,
        resume_command=f"expman resume {args.directory}",
    )


def _cmd_status(args: argparse.Namespace) -> int:
    if args.directory is None:
        return _live_overview()
    directory = _resolve_directory(args.directory)
    manifest = _read_manifest(directory)
    total_stages = len(manifest.get("pipeline") or [])
    queue = manifest.get("queue", [])
    active = manifest.get("active_gpu", {})
    progress = manifest.get("stage_progress", {})
    print(
        f"Batch {directory} | stages: {total_stages}"
        f" | runs: {len(manifest['experiments'])}"
    )
    for entry in manifest["experiments"]:
        run_id = entry.get("run_id", "?")
        attempts = entry.get("attempts", [])
        if run_id in active:
            state = "running"
        elif run_id in queue:
            state = "pending"
        elif attempts:
            state = str(attempts[-1]["status"]).lower()
        else:
            state = "unknown"
        stage = progress.get(run_id)
        position = (
            f"stage {stage + 1}/{total_stages}"
            if isinstance(stage, int) and total_stages
            else f"stage -/{total_stages}"
        )
        history = " ".join(str(item["status"]).lower() for item in attempts)
        print(f"  {run_id}  {state:<10} {position}  attempts: {history or 'none'}")
    elapsed, remaining = _read_timing(directory)
    print(f"elapsed: {elapsed} | remaining: {remaining}")
    url = _read_dashboard(directory)
    if url is not None:
        print(f"dashboard: {url}")
    return 0


def _overview_line(batch_id: int, directory: Path) -> str:
    """One summary line of the runs/ overview, tolerating live rewriting."""
    name = str(_RUNS_DIR / directory.name)
    try:
        manifest = read_record(directory / "batch.pkl", PickleSerializer())
    except (StorageError, OSError):
        manifest = None
    if not isinstance(manifest, dict):
        return f"{batch_id:>2}  {name:<30} unreadable records"
    experiments = manifest.get("experiments") or []
    queue = manifest.get("queue") or []
    active = manifest.get("active_gpu") or {}
    counts = {"run": 0, "pend": 0, "done": 0, "fail": 0, "stop": 0}
    for entry in experiments:
        run_id = entry.get("run_id", "?")
        attempts = entry.get("attempts") or []
        if run_id in active:
            counts["run"] += 1
        elif run_id in queue or not attempts:
            counts["pend"] += 1
        else:
            state = str(attempts[-1].get("status", "")).lower()
            if "succeed" in state:
                counts["done"] += 1
            elif "fail" in state:
                counts["fail"] += 1
            else:
                counts["stop"] += 1
    total_stages = len(manifest.get("pipeline") or [])
    reached = [
        value
        for value in (manifest.get("stage_progress") or {}).values()
        if isinstance(value, int)
    ]
    stage = (
        f"{max(reached) + 1}/{total_stages}"
        if reached and total_stages
        else f"-/{total_stages}"
    )
    elapsed, remaining = _read_timing(directory)
    return (
        f"{batch_id:>2}  {name:<30} {len(experiments):>4} "
        f"{counts['run']:>4} {counts['pend']:>4} {counts['done']:>4} "
        f"{counts['fail']:>4} {counts['stop']:>4}  "
        f"{stage:<8}{elapsed:<10}{remaining}"
    )


def _print_overview() -> None:
    ids = _ensure_ids()
    print(
        f"batches under {_RUNS_DIR}/"
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
    pid_path = directory / _PID_FILE
    try:
        pid = int(pid_path.read_text())
    except (FileNotFoundError, ValueError):
        print(f"{directory} is not running (no live PID record).", file=sys.stderr)
        return 1
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_path.unlink(missing_ok=True)
        print(
            f"process {pid} is gone; removed the stale PID record.",
            file=sys.stderr,
        )
        return 1
    except PermissionError:
        print(f"no permission to signal process {pid}.", file=sys.stderr)
        return 1
    print(f"Stop signal sent to {pid}; the batch persists and exits shortly.")
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
        command.add_argument(
            "--web",
            action="store_true",
            help="serve a read-only dashboard on loopback for this batch",
        )
        command.add_argument(
            "--web-port",
            type=int,
            default=DEFAULT_PORT,
            help="dashboard port (default: 8765; 0 picks a free one)",
        )
        command.add_argument(
            "--web-host",
            default="127.0.0.1",
            help="dashboard bind address (default: loopback only)",
        )
    run.set_defaults(handler=_cmd_run)
    resume.set_defaults(handler=_cmd_resume)

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
