"""Memory-gated per-GPU process scheduling and durable attempt collection."""

import os
import pickle
import signal
import subprocess
import sys
import warnings
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter, sleep

from . import devices
from .events import Status
from .experiment import AttemptResult, ResultOutput
from .storage import PickleSerializer, RecoveryWarning, read_record, write_record


@dataclass
class Gate:
    """One card's admission, held while its VRAM reading stays tight."""

    gpu_block: bool = False

    def can_launch(self, ratio, running):
        return ratio >= devices.MEMORY_MARGIN and (not self.gpu_block or not running)


@dataclass
class HostGate:
    """Admission for host RAM, which every worker process shares."""

    mem_block: bool = False

    def can_launch(self, host, running):
        if host is None or host.tight:
            return False
        return not self.mem_block or not running


@dataclass
class Worker:
    process: subprocess.Popen
    experiment: object
    device: int
    uuid: str
    root: Path
    started: float
    area: float = 0.0
    offset: int = 0
    # Sampled every tick: resident is what the worker holds now, peak is the most
    # it has ever held (VmHWM). Admission reserves the gap between the two.
    resident_kb: float = 0.0
    peak_kb: float = 0.0
    buffer: bytes = field(default=b"", repr=False)


def _kill_group(process):
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)


class GpuScheduler:
    def __init__(self, batch):
        if os.name != "posix":
            raise RuntimeError("GPU process scheduling currently requires Linux or WSL")
        self.batch = batch
        self.monitor = devices.NvidiaMemory(batch.devices)
        self.host = devices.MeminfoMonitor()
        self.gates = {device: Gate() for device in batch.devices}
        self.host_gate = HostGate()
        self.workers = {}
        self.last_tick = perf_counter()
        # Peak resident memory per run, from attempts that already ran. Admission
        # reserves the sum of these, so an unobserved run is charged the largest
        # peak any run has reported.
        self.peaks = {}
        for item in batch._gpu_history:
            peak = item.get("peak_kb")
            if peak:
                self.peaks[item["run_id"]] = max(
                    self.peaks.get(item["run_id"], 0.0), peak
                )
        self.peak_ceiling = max(self.peaks.values(), default=0.0)

    def _account(self):
        now = perf_counter()
        for worker in self.workers.values():
            count = sum(item.device == worker.device for item in self.workers.values())
            worker.area += (now - max(self.last_tick, worker.started)) * count
            residency = devices.residency_kb(worker.process.pid)
            if residency is not None:
                worker.resident_kb, worker.peak_kb = residency
        self.last_tick = now
        with self.batch._state_lock:
            self.batch._gpu_running_info = {
                run_id: {
                    "device": worker.device,
                    "uuid": worker.uuid,
                    "concurrency": worker.area / max(now - worker.started, 1e-9),
                    "duration": max(0.0, now - worker.started),
                    "resident_kb": worker.resident_kb,
                    "peak_kb": worker.peak_kb,
                }
                for run_id, worker in self.workers.items()
            }

    def _events(self, worker):
        path = worker.root / "events.bin"
        if not path.exists():
            return
        with path.open("rb") as stream:
            stream.seek(worker.offset)
            data = stream.read(1024 * 1024)
        worker.offset += len(data)
        worker.buffer += data
        while len(worker.buffer) >= 8:
            length = int.from_bytes(worker.buffer[:8], "big")
            if len(worker.buffer) < length + 8:
                break
            payload, worker.buffer = (
                worker.buffer[8 : 8 + length],
                worker.buffer[8 + length :],
            )
            try:
                self.batch.recorder.record(pickle.loads(payload))
            except Exception as error:
                warnings.warn(f"recorder failed: {error}", RuntimeWarning, stacklevel=2)

    def _observe(self):
        """Read device and host memory; a failed or timed-out query is not a reading."""
        try:
            memory = self.monitor.sample()
            host = self.host.sample()
        except devices.MemoryObservationError as error:
            warnings.warn(
                f"memory query failed; treating host memory as tight: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            with self.batch._state_lock:
                self.batch._gpu_memory = {}
                self.batch._host_memory = {}
            return None, None
        admits_next = self._admits(
            host, self.peak_ceiling or devices.HOST_PEAK_KB_DEFAULT
        )
        with self.batch._state_lock:
            self.batch._gpu_memory = {
                device: value.free_ratio for device, value in memory.items()
            }
            self.batch._host_memory = {
                "available_ratio": host.available_ratio,
                "available_kb": host.available_kb,
                "total_kb": host.total_kb,
                "headroom_kb": host.headroom_kb,
                "swap_total_kb": host.swap_total_kb,
                "swap_free_kb": host.swap_free_kb,
                "paging": host.paging,
                "tight": host.tight,
                "admits_next": admits_next,
            }
        return memory, host

    def _relieve(self, memory, host):
        """Hold blocked cards and shed the newest attempt of each tight resource."""
        if host is None:
            # Without a reading the host cannot be assumed to have room, so this
            # checks as host memory shortage: block launches and drop one attempt.
            self.host_gate.mem_block = True
            if self.workers:
                self._shed(self._newest())
            return
        if host.tight:
            # Host RAM is shared by every worker process: drop the newest attempt
            # globally and stop refilling any card until one of the survivors
            # exits on its own. One attempt per tick, not one decisive sweep: the
            # shed frees its memory by the next tick, so pressure that survives
            # the reserve keeps shedding until the reading recovers.
            self.host_gate.mem_block = True
            if self.workers:
                self._shed(self._newest())
        for device, gate in self.gates.items():
            if memory[device].free_ratio < devices.MEMORY_MARGIN:
                gate.gpu_block = True
                running = [
                    key
                    for key, worker in self.workers.items()
                    if worker.device == device
                ]
                if running:
                    self._shed(running[-1])

    def _newest(self):
        return max(self.workers, key=lambda run_id: self.workers[run_id].started)

    def _shed(self, run_id):
        """Cancel one running attempt for memory pressure on its card."""
        self._finish(run_id, cancelled="MemoryPressure")

    def _expected_peak(self, run_id):
        """The largest resident set this run has been seen to need, in kilobytes.

        An unobserved run is charged the largest peak any run has reported: the
        reserve has to upper-bound the next attempt, and only throughput suffers
        when the bound is too generous.
        """
        return (
            self.peaks.get(run_id) or self.peak_ceiling or devices.HOST_PEAK_KB_DEFAULT
        )

    def _admits(self, host, expected_kb):
        """Whether one more attempt of expected_kb fits above the host reserve.

        Every running worker is charged for the gap between what it holds now and
        the most it has been seen to need, so a launch cannot spend memory that
        attempts already in flight are still going to ask for.
        """
        needed = expected_kb
        for run_id, worker in self.workers.items():
            reach = max(worker.peak_kb, self._expected_peak(run_id))
            needed += max(0.0, reach - worker.resident_kb)
        return devices.fits_reserve(host, needed)

    def _launch(self, device, memory, host):
        batch = self.batch
        with batch._state_lock:
            run_id = batch._next_experiment()
        if not self._admits(host, self._expected_peak(run_id)):
            # Put the selection back at the front. Dispatching updated the
            # estimator's own state, and re-selecting the same run is idempotent,
            # so the queue keeps its remaining order while the host is short.
            with batch._state_lock:
                batch._queue.appendleft(run_id)
            return False
        with batch._state_lock:
            batch._active_gpu[run_id] = device
        experiment = next(item for item in batch.experiments if item.run_id == run_id)
        attempt = len(experiment.result.attempts) + 1
        root = experiment.output_dir / "attempts" / str(attempt)
        process = None
        try:
            root.mkdir(parents=True, exist_ok=True)
            main_module = sys.modules.get("__main__")
            needs_main = any(
                cls.__module__ == "__main__"
                for cls in (*experiment.pipeline.stages, type(batch._serializer))
            )
            script = getattr(main_module, "__file__", None) if needs_main else None
            if needs_main and (script is None or not Path(script).is_file()):
                raise ValueError(
                    "GPU stages must be importable; run from a Python script"
                )
            write_record(
                root / "bootstrap.pkl",
                {
                    "main_script": str(Path(script).resolve()) if script else None,
                    "sys_path": [str(Path(path).resolve()) for path in sys.path],
                },
                PickleSerializer(),
            )
            write_record(
                root / "input.pkl",
                {
                    "pipeline": experiment.pipeline,
                    "cfg": experiment.cfg,
                    "run_id": run_id,
                    "output_dir": experiment.output_dir,
                    "serializer": batch._serializer,
                    "metrics_path": experiment._metrics.path,
                    "cache_root": experiment._cache_root,
                    "attempts": [
                        replace(item, _output=None, output_source=None)
                        for item in experiment.result.attempts
                    ],
                },
                PickleSerializer(),
            )
            batch._save()  # Durable active membership precedes process creation.
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = memory.uuid
            env["PYTHONPATH"] = os.pathsep.join(
                str(Path(path).resolve()) for path in sys.path
            )
            self._account()
            with (root / "output.log").open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", "expman._worker", str(root)],
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            now = perf_counter()
            with experiment._timing_lock:
                experiment._active_started = now
            self.workers[run_id] = Worker(
                process, experiment, device, memory.uuid, root, now
            )
        except BaseException:
            if process is not None:
                _kill_group(process)
                process.wait()
                process.stdin.close()
                self.workers.pop(run_id, None)
                with experiment._timing_lock:
                    experiment._active_started = None
            with batch._state_lock:
                batch._active_gpu.pop(run_id, None)
                batch._queue.appendleft(run_id)
            batch._save()
            raise
        return True

    def _finish(self, run_id, *, cancelled=None):
        worker = self.workers[run_id]
        self._account()
        finished = perf_counter()
        _kill_group(worker.process)  # Also reap experiment-owned helper processes.
        worker.process.wait()
        worker.process.stdin.close()
        while (worker.root / "events.bin").exists():
            before = worker.offset
            self._events(worker)
            if before == worker.offset:
                break
        experiment = worker.experiment
        duration = max(0.0, finished - worker.started)
        result_path = worker.root / "result.pkl"
        if result_path.exists():
            result = read_record(result_path, self.batch._serializer)
            if (
                not isinstance(result, AttemptResult)
                or result.run_id != run_id
                or result.attempt != len(experiment.result.attempts) + 1
            ):
                raise RuntimeError("invalid worker attempt result")
            result = replace(result, duration_seconds=duration)
            # The record stays authoritative for the output: keep only the summary
            # so the parent does not hold one result object per completed attempt.
            result = replace(
                result,
                _output=None,
                output_source=ResultOutput(result_path, self.batch._serializer),
            )
        else:
            result = AttemptResult(
                run_id,
                len(experiment.result.attempts) + 1,
                Status.CANCELLED if cancelled else Status.FAILED,
                duration,
                error_type=cancelled or "WorkerExit",
                error_message=f"worker exited with code {worker.process.returncode}",
            )
        with self.batch._state_lock:
            with experiment._timing_lock:
                experiment._attempts.append(result)
                experiment._active_started = None
            self.batch._gpu_history.append(
                {
                    "run_id": run_id,
                    "attempt": result.attempt,
                    "device": worker.device,
                    "uuid": worker.uuid,
                    "concurrency": worker.area / max(duration, 1e-9),
                    "duration": duration,
                    "peak_kb": worker.peak_kb,
                }
            )
            if worker.peak_kb:
                self.peaks[run_id] = max(self.peaks.get(run_id, 0.0), worker.peak_kb)
                self.peak_ceiling = max(self.peak_ceiling, worker.peak_kb)
            self.batch._active_gpu.pop(run_id)
            self.batch._gpu_running_info.pop(run_id, None)
            del self.workers[run_id]
            failures = sum(
                item.status is Status.FAILED for item in experiment.result.attempts
            )
            if result.status is Status.CANCELLED or (
                result.status is Status.FAILED and failures <= self.batch.max_retries
            ):
                self.batch._queue.append(run_id)
            if cancelled != "MemoryPressure":
                # A natural exit frees memory: release the host block and the block
                # of the card the attempt ran on.
                self.gates[worker.device].gpu_block = False
                self.host_gate.mem_block = False
            self.batch._save()

    def run(self):
        try:
            while self.batch._queue or self.workers:
                for run_id, worker in list(self.workers.items()):
                    self._events(worker)
                    if worker.process.poll() is not None:
                        self._finish(run_id)
                if not self.batch._queue and not self.workers:
                    break
                self._account()
                # Both readings are rechecked every tick: a stale ratio never
                # admits work, and an unreadable one keeps every card closed.
                memory, host = self._observe()
                self._relieve(memory, host)
                launching = self.host_gate.can_launch(host, list(self.workers))
                for device, gate in self.gates.items():
                    if memory is None or (
                        memory[device].free_ratio < devices.MEMORY_MARGIN
                    ):
                        continue
                    running = [
                        key
                        for key, worker in self.workers.items()
                        if worker.device == device
                    ]
                    if (
                        self.batch._queue
                        and launching
                        and gate.can_launch(memory[device].free_ratio, running)
                        and not self._launch(device, memory[device], host)
                    ):
                        # Host RAM is shared, so one refused launch closes every
                        # card for this tick.
                        break
                sleep(devices.POLL_INTERVAL)
            return self.batch.results
        except BaseException:
            # Kill every worker before doing any potentially slow IO or recovery work.
            for worker in self.workers.values():
                _kill_group(worker.process)
            for run_id in list(self.workers):
                try:
                    self._finish(run_id, cancelled="SchedulerInterrupted")
                except Exception as error:
                    warnings.warn(
                        f"could not record worker interruption: {error}",
                        RecoveryWarning,
                        stacklevel=2,
                    )
            raise
