"""Start a batch in its own session and report when it owns the stop endpoint."""

import json
import os
import shutil
import socket
import subprocess
import sys
from tempfile import TemporaryFile
from threading import Thread

from expman.cli.commands import build_parser
from expman.core.events import Status
from expman.settings import set as set_settings
from expman.settings import settings


def launch_background(argv: list[str]) -> int:
    """Wait for startup without tying the scheduler to the caller's terminal."""
    caller, scheduler = socket.socketpair()
    with (
        caller,
        scheduler,
        TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as diagnostics,
    ):
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "expman.cli._background",
                str(scheduler.fileno()),
                *argv,
            ],
            stdin=subprocess.DEVNULL,
            stdout=diagnostics,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=(scheduler.fileno(),),
        )
        scheduler.close()
        try:
            # Use the settings and import path visible to this CLI invocation.
            startup = {"settings": vars(settings), "sys_path": sys.path}
            caller.sendall(json.dumps(startup, default=str).encode())
            caller.shutdown(socket.SHUT_WR)
            with caller.makefile("r", encoding="utf-8") as ready:
                message = ready.readline()
        except (BrokenPipeError, ConnectionResetError):
            message = ""
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        if not message:
            # The interpreter may still be writing the startup traceback.
            status = process.wait()
            diagnostics.seek(0)
            shutil.copyfileobj(diagnostics, sys.stderr)
            print(f"Batch exited before startup (exit code {status}).", file=sys.stderr)
            return status if status > 0 else 1

        batch_id, directory = json.loads(message)
        print(f"Batch ID: {batch_id}\nBatch directory: {directory}", flush=True)
        # Embedded CLI callers may outlive their scheduler.
        Thread(target=process.wait, daemon=True).start()
        return 0


def _run(fd: int, argv: list[str]) -> int:
    with socket.socket(fileno=fd) as caller:
        caller.set_inheritable(False)
        with caller.makefile("r", encoding="utf-8") as request:
            startup = json.load(request)
        sys.path[:] = startup["sys_path"]
        set_settings(**startup["settings"])
        args = build_parser().parse_args(argv)
        batch_id, batch = args.handler(args)

        def started() -> None:
            sys.stdout.flush()
            sys.stderr.flush()
            with (
                os.fdopen(os.dup(1), "rb") as startup_log,
                (batch.output_dir / "batch.log").open("ab") as log,
            ):
                os.dup2(log.fileno(), 1)
                os.dup2(log.fileno(), 2)
                caller.sendall(
                    (json.dumps([batch_id, str(batch.output_dir)]) + "\n").encode()
                )
                caller.close()
                # Copy the startup output after releasing the terminal. pread
                # leaves the file position of any inherited writer untouched.
                size = os.fstat(startup_log.fileno()).st_size
                offset = 0
                while offset < size:
                    chunk = os.pread(
                        startup_log.fileno(), min(size - offset, 65536), offset
                    )
                    if not chunk:
                        break
                    log.write(chunk)
                    offset += len(chunk)

        try:
            results = batch.run(on_started=started)
        except KeyboardInterrupt:
            print("Batch stopped.", file=sys.stderr)
            return 143
        for result in results:
            print(f"{result.run_id}: {result.status.value}")
        return int(any(result.status is Status.FAILED for result in results))


if __name__ == "__main__":
    sys.exit(_run(int(sys.argv[1]), sys.argv[2:]))
