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

from . import devices, limits
from ._memory_model import PeakEstimator
from .events import Status
from .experiment import AttemptResult, ResultOutput
from .storage import PickleSerializer, RecoveryWarning, read_record, write_record


@dataclass
class Gate:
    """One card's admission, held after a CUDA out-of-memory failure.

    The block stops *adding* attempts to a card that already ran out of device
    memory while it still has attempts running; an empty card may try again, which
    is also what keeps a single failed attempt from stalling the Batch forever.
    """

    gpu_block: bool = False

    def can_launch(self, ratio, running):
        # ``ratio`` is retained for API compatibility only. There is no
        # percentage-based VRAM admission threshold anymore.
        return not self.gpu_block or not running


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
    limit: object = None
    # Sampled every tick: resident is what the worker holds now, peak is the most
    # it has ever held. Admission reserves the gap between the two.
    resident_kb: float = 0.0
    peak_kb: float = 0.0
    cap_hit: bool = False
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
        # Peak host memory per attempt: admission charges the configuration-based
        # estimate and the cgroup cap enforces it, so the scheduler keeps no
        # per-run table of its own here.
        self.peaks = PeakEstimator.from_batch(batch)

    @property
    def peak_ceiling(self):
        """Largest observed host peak, retained for scheduler introspection."""
        return self.peaks.ceiling_kb

    def _expected_peak(self, run_id):
        """Return the feature-based reservation for one pending run."""
        return self.peaks.estimate_kb(run_id)

    def _account(self):
        now = perf_counter()
        for worker in self.workers.values():
            count = sum(item.device == worker.device for item in self.workers.values())
            worker.area += (now - max(self.last_tick, worker.started)) * count
            residency = limits.usage(worker.limit.cgroup if worker.limit else None)
            if residency is None:
                residency = devices.residency_kb(worker.process.pid)
            if residency is not None:
                worker.resident_kb, worker.peak_kb = residency
            if worker.limit is not None and limits.refused(
                limits.events(worker.limit.cgroup)
            ):
                worker.cap_hit = True
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
        # One more attempt is charged at least the largest peak seen so far.
        admits_next = self._admits(
            host, max(self.peaks.ceiling_kb, devices.HOST_PEAK_KB_DEFAULT)
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

    def _newest(self):
        return max(self.workers, key=lambda run_id: self.workers[run_id].started)

    def _cap_report(self, worker):
        """(limit refused a charge, peak kilobytes) for one finished worker.

        The worker writes the record from inside its own cgroup, because a cgroup
        OOM kill leaves the parent with nothing but an unexplained signal; the
        parent's own sampling covers a record that never got written.
        """
        capped = worker.cap_hit
        peak = worker.peak_kb
        path = worker.root / "memory_cap.pkl"
        if path.exists():
            report = read_record(path, PickleSerializer())
            if isinstance(report, dict):
                capped = capped or limits.refused(report.get("events") or {})
                recorded = report.get("usage")
                if isinstance(recorded, (list, tuple)) and len(recorded) == 2:
                    peak = max(peak, float(recorded[1]))
        if not capped and worker.limit is not None:
            # A peak that reached the limit, or a SIGKILL with no recorded cause,
            # is this attempt's cap rather than an unrelated crash: the counters on
            # the kill path can miss the window, and the remedy is the same.
            capped = limits.near_cap(peak, worker.limit.cap_kb) or (
                worker.process.returncode == -signal.SIGKILL
            )
        return capped, peak

    def _shed(self, run_id):
        """Cancel one running attempt for memory pressure on its card."""
        self._finish(run_id, cancelled="MemoryPressure")

    def _admits(self, host, expected_kb):
        """Whether one more attempt of expected_kb fits above the host reserve.

        Every running worker is charged for the gap between what it holds now and
        the most it has been seen to need, so a launch cannot spend memory that
        attempts already in flight are still going to ask for.
        """
        needed = expected_kb
        for run_id, worker in self.workers.items():
            reach = max(worker.peak_kb, self.peaks.estimate_kb(run_id))
            needed += max(0.0, reach - worker.resident_kb)
        return devices.fits_reserve(host, needed)

    def _launch(self, device, memory, host):
        batch = self.batch
        with batch._state_lock:
            run_id = batch._next_experiment()
        if not self._admits(host, self.peaks.estimate_kb(run_id)):
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
        limit = None
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
            # The unit is unique per attempt: systemd unloads a finished scope
            # asynchronously, and a retry must not collide with the previous one.
            limit = limits.memory_limit(
                f"expman-{run_id[:12]}-{attempt}", self.peaks.estimate_kb(run_id)
            )
            if limit.cgroup is not None:
                env["EXPMAN_CGROUP"] = str(limit.cgroup)
            self._account()
            with (root / "output.log").open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    [*limit.command, sys.executable, "-m", "expman._worker", str(root)],
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            if (
                limit.mechanism == "cgroup"
                and limit.cgroup is not None
                and not limits.join(limit.cgroup, process.pid)
            ):
                raise RuntimeError(
                    f"could not move worker {process.pid} into {limit.cgroup}"
                )
            now = perf_counter()
            with experiment._timing_lock:
                experiment._active_started = now
            self.workers[run_id] = Worker(
                process, experiment, device, memory.uuid, root, now, limit=limit
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
        capped, peak_kb = self._cap_report(worker)
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
        if capped and cancelled is None and result.status is not Status.SUCCEEDED:
            # The limit, not the experiment, ended this attempt: it is retried with
            # a raised peak estimate instead of spending the failure budget, and it
            # carries the peak it reached as a lower bound for that estimate. An
            # attempt that finished its work inside a throttled limit still counts
            # as done; the raised estimate only affects the attempts after it.
            result = replace(
                result,
                status=Status.CANCELLED,
                error_type="MemoryLimit",
                error_message=(
                    "attempt stopped by its memory limit of "
                    f"{worker.limit.cap_kb / 1024:.0f} MiB"
                    if worker.limit is not None
                    else "attempt stopped by its memory limit"
                ),
            )
        with self.batch._state_lock:
            with experiment._timing_lock:
                experiment._attempts.append(result)
                experiment._active_started = None
            observation = {
                "run_id": run_id,
                "attempt": result.attempt,
                "device": worker.device,
                "uuid": worker.uuid,
                "concurrency": worker.area / max(duration, 1e-9),
                "duration": duration,
                "peak_kb": peak_kb,
                "capped": capped,
            }
            self.batch._gpu_history.append(observation)
            self.peaks.record(observation)
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
            out_of_memory = devices.cuda_out_of_memory(
                result.error_type, result.error_message
            )
            if out_of_memory:
                # Device memory is only known to be short once a kernel failed.
                self.gates[worker.device].gpu_block = True
            if cancelled != "MemoryPressure" and not capped:
                # A natural exit frees memory: release the host block, and release
                # the card's block unless the attempt itself ran out of device
                # memory (that block is lifted by an attempt that ends normally).
                self.host_gate.mem_block = False
                if not out_of_memory and (
                    result.status is Status.SUCCEEDED or result.status is Status.FAILED
                ):
                    self.gates[worker.device].gpu_block = False
            self.batch._gpu_blocks = {
                device: gate.gpu_block for device, gate in self.gates.items()
            }
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
                    if memory is None:
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
