"""Best-effort CLI progress for a running batch."""

import logging
import math
import os
import sys
import unicodedata
from numbers import Real
from threading import Event, Thread
from time import monotonic

from .events import Status

_logger = logging.getLogger(__name__)
_UNKNOWN = "??:??:??～??:??:??"


class BatchProgress:
    def __init__(self, batch, *, enabled: bool, interval: float):
        if not isinstance(enabled, bool):
            raise TypeError("progress must be a boolean")
        if (
            isinstance(interval, bool)
            or not isinstance(interval, Real)
            or not math.isfinite(interval)
            or interval <= 0
        ):
            raise ValueError("refresh_interval must be a positive finite number")
        self.batch = batch
        self.enabled = enabled
        self.interval = interval
        self.stream = sys.stderr
        self.stopped = Event()
        self.thread = None
        self.disabled = False
        self.warned = False
        self.last_line = None
        self.inline = False
        self.started_at = None
        self.finished_at = None
        try:
            self.tty = bool(self.stream.isatty())
        except Exception:
            self.tty = False

    def start(self) -> None:
        if not self.enabled:
            return
        self.started_at = monotonic()
        self._render()
        if not self.disabled:
            self.thread = Thread(target=self._loop, name="expman-progress", daemon=True)
            self.thread.start()

    def stop(self, error: BaseException | None = None) -> None:
        if not self.enabled:
            return
        self.finished_at = monotonic()
        self.stopped.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join()
        label = "已结束"
        if error is not None:
            label = "已中断" if not isinstance(error, Exception) else "执行异常"
        self._render(label=label, final=True)

    def _loop(self) -> None:
        while not self.stopped.wait(self.interval):
            self._render()

    def _snapshot(self):
        with self.batch._state_lock:
            pending = set(self.batch._queue)
            pending.update(self.batch._active_gpu)
            succeeded = failed = 0
            current = None
            for index, experiment in enumerate(self.batch.experiments, start=1):
                attempts, _ = experiment._timing_snapshot()
                status = attempts[-1].status if attempts else Status.PENDING
                succeeded += status is Status.SUCCEEDED
                failed += status is Status.FAILED and experiment.run_id not in pending
                if experiment.run_id in self.batch._active_gpu:
                    current = index
            cards = " ".join(
                f"GPU{device}:{sum(value == device for value in self.batch._active_gpu.values())}运行"
                for device in self.batch.devices
            )
            return len(self.batch.experiments), succeeded, failed, current, cards

    def _render(self, *, label="运行中", final=False) -> None:
        if self.disabled:
            return
        try:
            total, succeeded, failed, current, cards = self._snapshot()
            try:
                remaining = str(self.batch._display_estimate())
            except Exception:
                remaining = _UNKNOWN
                if not self.warned:
                    self.warned = True
                    _logger.warning("Could not estimate remaining time", exc_info=True)
            # A slow estimate can outlive the batch. Discard that stale frame before
            # writing the final snapshot from the execution thread.
            if self.stopped.is_set() and not final:
                return
            completed = succeeded + failed
            percent = 100 if total == 0 else completed * 100 // total
            position = f" | 当前 {current}/{total}" if current is not None else ""
            if cards:
                position += f" | {cards}"
            minutes = int(max(0, self.batch.elapsed_seconds) // 60)
            days, minutes = divmod(minutes, 24 * 60)
            hours, minutes = divmod(minutes, 60)
            elapsed = f"{days:02d}:{hours:02d}:{minutes:02d}"
            line = (
                f"{label} {completed}/{total} ({percent}%)"
                f" | 成功 {succeeded} 失败 {failed}{position}"
                f" | 已运行 {elapsed} | 剩余 {remaining}"
            )
            if not self.tty and line == self.last_line and not final:
                return
            self._write(line, final)
            self.last_line = line
        except Exception:
            # Presentation failures must not fail or retry a business experiment.
            self.disabled = True
            self.stopped.set()

    def _write(self, line, final):
        width = sum(
            0
            if unicodedata.combining(char)
            else 2
            if unicodedata.east_asian_width(char) in "WF"
            else 1
            for char in line
        )
        try:
            columns = os.get_terminal_size(self.stream.fileno()).columns
        except (AttributeError, OSError, ValueError):
            columns = 80
        inline = self.tty and width < columns
        prefix = "\r\x1b[2K" if self.inline or inline else ""
        suffix = "\n" if final or not inline else ""
        self.stream.write(prefix + line + suffix)
        self.stream.flush()
        self.inline = inline and not final
