"""Memory-gated per-GPU process scheduling and durable attempt collection."""

import os
import signal
import socket
import subprocess
import sys
import warnings
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter, sleep, time

import expman.resources.devices as devices
import expman.resources.limits as limits
import expman.scheduling.environment as environment
from expman.core.dependencies import stage_dependencies
from expman.core.events import Status
from expman.core.experiment import StageResult
from expman.estimation.memory_model import PeakCeiling
from expman.resources.mps import MpsService
from expman.scheduling._planning import pack_candidates
from expman.scheduling._prefixes import PrefixReclaimer
from expman.scheduling._process import (
    Worker,
    _close_stdin,
    _kill_group,
)
from expman.scheduling._resource import (
    _admits,
    _admits_gpu,
    _device_unbacked,
    _resource_left,
)
from expman.scheduling._stages import StageWork
from expman.scheduling.ipc import Channel
from expman.settings import settings
from expman.storage.base import PickleSerializer, RecoveryWarning, write_record


class GpuScheduler:
    def __init__(self, batch):
        if os.name != "posix":
            raise RuntimeError("GPU process scheduling currently requires Linux or WSL")
        self.batch = batch
        self.monitor = devices.NvidiaMemory(batch.devices)
        self.host = devices.MeminfoMonitor()
        # Re-read the worker environment at the start of each execution session.
        self._environment = environment.overlay()
        self.mps = MpsService()
        self.workers = {}
        self.last_tick = perf_counter()
        # The ETA slot forecast uses host peaks from measured final Stages.
        self.stages = StageWork(batch)
        self.prefixes = PrefixReclaimer(self.stages)
        last_stage = len(batch.pipeline.stages) - 1
        self.peaks = PeakCeiling.from_history(
            row
            for row in batch.state.stage_history
            if row.get("stage") == last_stage
            and row.get("status") == Status.SUCCEEDED.value
            and not row.get("reused")
        )
        # Dashboard state: a bounded event log and the admission plan this
        # scheduler actually chose.  Both are published observations for the
        # dashboard to read, and neither is ever an input to a decision.
        self.events = deque(maxlen=200)
        self.last_plan = None
        # Shedding under pressure is rate limited, and the deadline is the only
        # state it needs: the memory an ended attempt held is not visible to the
        # reading that follows it.
        self._relief_until = 0.0
        batch._scheduler = self

    def _note(self, kind, **fields):
        """Record one scheduling event for the dashboard; never a decision input."""
        self.events.append({"at": time(), "kind": kind, **fields})

    def _account(self):
        now = perf_counter()
        for worker in self.workers.values():
            count = sum(item.device == worker.device for item in self.workers.values())
            worker.area += (now - max(self.last_tick, worker.started)) * count
            if limits.refused(worker.memory_events):
                worker.cap_hit = True
        self.last_tick = now
        with self.batch._state_lock:
            self.batch.live.running_info = {
                run_id: {
                    "device": worker.device,
                    "uuid": worker.uuid,
                    "concurrency": worker.area / max(now - worker.started, 1e-9),
                    "duration": max(0.0, now - worker.started),
                    "progress_forecast": worker.progress_forecast,
                    "stage_index": worker.stage_index,
                    "resident_kb": worker.resident_kb,
                    "peak_kb": worker.peak_kb,
                    "gpu_used_kb": worker.gpu_used_kb,
                    "gpu_process_peak_kb": worker.gpu_process_peak_kb,
                }
                for run_id, worker in self.workers.items()
            }

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
                self.batch.live.host_memory = {"admits_next": False}
                self.batch.live.device_memory = {}
            return None, None
        # ETA opens idle slots when this conservative host estimate fits.
        # Actual launches use each candidate's own per-Stage estimate.
        admits_next = _admits(
            self.workers,
            host,
            max(self.peaks.ceiling_kb, settings.host_peak_kb_default),
        )
        with self.batch._state_lock:
            self.batch.live.host_memory = {
                "available_ratio": host.available_ratio,
                "available_kb": host.available_kb,
                "total_kb": host.total_kb,
                "headroom_kb": host.headroom_kb,
                "swap_total_kb": host.swap_total_kb,
                "swap_free_kb": host.swap_free_kb,
                "tight": host.tight,
                "admits_next": admits_next,
                "observed_at": time(),
            }
            # Device readings are published alongside the host reading: the
            # dashboard shows allocation against actual usage, and nothing else
            # keeps the per-card totals the scheduler just queried.
            self.batch.live.device_memory = {
                device: {
                    "uuid": observation.uuid,
                    "total_kb": observation.total * 1024,
                    "free_kb": observation.free * 1024,
                    "observed_at": time(),
                }
                for device, observation in memory.items()
            }
        return memory, host

    def _relieve(self, memory, host):
        """Shed one attempt when reservations already handed out are unbacked.

        Shedding resolves over-commitment between attempts sharing a machine.
        With one attempt running there is nothing to distribute: ending it
        discards a cold start no other attempt is competing with, while that
        worker's own cgroup and MPS client limit govern it. A
        shortage with one worker therefore pauses launches and keeps the attempt.

        The host check is admission's own arithmetic with nothing new to place:
        if one more attempt of zero size would not fit, what the host cannot
        honour is the reservations already running. Pressure that is not the
        host's is looked for on each card in turn. One attempt per shedding,
        and none while a previous one is still being released.
        """
        if len(self.workers) < 2 or perf_counter() < self._relief_until:
            return
        if host is None or not _admits(self.workers, host, 0.0):
            kind, device = "host", None
        else:
            device = next(
                (
                    item
                    for item in (memory or ())
                    if sum(w.device == item for w in self.workers.values()) > 1
                    and _device_unbacked(self.workers, memory, item)
                ),
                None,
            )
            if device is None:
                return
            kind = "device"
        # The attempt launched last is the one whose reservation this machine
        # has had least time to honour, and the cheapest to give back.
        self._shed(self._newest(device), kind, device)
        self._relief_until = perf_counter() + settings.relief_cooldown

    def _newest(self, device=None):
        """The attempt launched last, globally or on one card."""
        running = self.workers
        if device is not None:
            on_card = {
                run_id: worker
                for run_id, worker in running.items()
                if worker.device == device
            }
            if on_card:
                running = on_card
        return max(running, key=lambda run_id: running[run_id].started)

    def _cap_report(self, worker):
        """Return the cgroup result last reported by the worker over IPC."""
        exit_events = {} if worker.limit is None else limits.events(worker.limit.cgroup)
        capped = (
            worker.cap_hit
            or limits.refused(worker.memory_events)
            or limits.refused(exit_events)
        )
        peak = worker.peak_kb
        if not capped and worker.limit is not None and worker.limit.capped:
            # A peak that reached the limit, or a SIGKILL with no recorded cause,
            # is this attempt's cap rather than an unrelated crash: the counters on
            # the kill path can miss the window, and the remedy is the same.
            effective_cap = worker.host_limit_kb or worker.limit.cap_kb
            capped = limits.near_cap(peak, effective_cap) or (
                worker.process.returncode == -signal.SIGKILL
            )
        return capped, peak

    def _shed(self, run_id, kind, device=None):
        """Cancel one running attempt whose reservation this machine cannot honour."""
        worker = self.workers[run_id]
        self._note(
            "shed",
            run_id=run_id,
            stage_index=worker.stage_index,
            stage_attempt=worker.stage_attempt,
            device=worker.device,
            reason=kind,
            detail=(
                "主机内存不足以兑现已分配的配额，撤销最新的尝试"
                if kind == "host"
                else f"设备 {device} 显存不足以兑现已分配的配额，撤销最新的尝试"
            ),
        )
        self._finish(run_id, cancelled="MemoryPressure")

    def _launch_plan(self, memory, host):
        """Choose a bounded multi-resource packing plan for this scheduler tick."""
        host_capacity, available_gpu = _resource_left(self.workers, memory, host)
        unknown = {w.device for w in self.workers.values() if w.gpu_used_kb is None}
        devices_in_plan = tuple(device for device in memory if device not in unknown)
        for device in unknown.intersection(available_gpu):
            available_gpu[device] = 0.0
        candidates = None
        placements = ()
        host_fits = False if host_capacity <= 0 else None
        gpu_fits = None
        if host_capacity > 0 and devices_in_plan:
            gpu_capacities = tuple(
                max(0.0, available_gpu[device]) for device in devices_in_plan
            )
            gpu_totals = tuple(
                memory[device].total * 1024 for device in devices_in_plan
            )
            # A Stage may need a larger card even when a smaller one is free.
            largest_gpu_kb = max(card.total for card in memory.values()) * 1024
            candidates = self.stages.ready_candidates(largest_gpu_kb, self.workers)
            plan = pack_candidates(
                candidates, host_capacity, devices_in_plan, gpu_capacities, gpu_totals
            )
            placements = tuple(
                (replace(c, gpu_cap_kb=min(c.gpu_cap_kb, memory[d].total * 1024)), d)
                for c, d in plan
            )
            if candidates:
                host_fits = any(c.host_reserve_kb <= host_capacity for c in candidates)
                gpu_fits = any(
                    c.gpu_reserve_kb <= max(gpu_capacities) for c in candidates
                )
        self.last_plan = {
            "at": time(),
            "ready": len(candidates) if candidates is not None else None,
            "host_fits": host_fits,
            "gpu_fits": gpu_fits,
            "capacity": {
                "host_left_kb": host_capacity,
                "gpu_left_kb": available_gpu,
            },
            "placements": [
                {
                    "run_id": candidate.run_id,
                    "stage_index": candidate.stage_index,
                    "stage": self.batch.pipeline.stages[candidate.stage_index].__name__,
                    "device": device,
                    "host_reserve_kb": candidate.host_reserve_kb,
                    "gpu_reserve_kb": candidate.gpu_reserve_kb,
                    "gpu_cap_kb": candidate.gpu_cap_kb,
                }
                for candidate, device in placements
            ],
            "launched": [],
        }
        return placements

    def _launch_available(self, memory, host):
        """Launch the resource-aware packing plan chosen from this tick's snapshot."""
        placements = self._launch_plan(memory, host)
        launched = [
            candidate.run_id
            for candidate, device in placements
            if self._launch(device, memory[device], host, candidate)
        ]
        if self.last_plan is not None:
            self.last_plan = {**self.last_plan, "launched": launched}

    def _complete_stage(self, run_id, stage_index, observation=None):
        """Advance a successful run or record its completion."""
        if stage_index < len(self.batch.pipeline.stages) - 1:
            self.batch.state.advance(run_id)
            return
        self.batch.record_attempt(run_id, Status.SUCCEEDED)
        if observation is not None and not observation["reused"]:
            self.peaks.record(observation)

    def _reuse_stage(self, run_id):
        """Materialize one pending Stage from shared metadata when possible."""
        experiment = self.batch.by_id[run_id]
        stage_index = self.batch.state.stage_progress[run_id]
        store = experiment._store
        shared = store.shared
        if shared is None or not store.can_reuse((stage_index,)):
            return False
        parent = "root"
        if stage_index:
            parent_ref = store.completed_reference((stage_index - 1,))
            if parent_ref is None:
                return False
            parent = parent_ref[2]
        # Reuse needs only metadata. A downstream worker loads its input
        # from the intermediate snapshot when it runs.
        stage = experiment.pipeline.stages[stage_index]
        dependencies = stage_dependencies(stage, experiment._frozen)
        candidate = shared.find_metadata(
            parent,
            (stage_index,),
            dependencies,
            leaf=stage_index == len(experiment.pipeline.stages) - 1,
        )
        if candidate is None:
            return False
        reference, completed = candidate
        attempt = self.batch.state.next_attempt(run_id, stage_index)
        store.reuse((stage_index,), attempt, reference, completed)
        store.status(
            (stage_index,),
            Status.SUCCEEDED.value,
            attempt,
            reused=True,
            elapsed_seconds=completed["elapsed_seconds"],
            restore_seconds=0.0,
        )
        with self.batch._state_lock:
            self.batch.state.reuse(run_id, stage_index)
            timestamp = round(self.batch.elapsed_seconds, 2)
            self.batch.state.stage_history.append(
                {
                    "run_id": run_id,
                    "stage": stage_index,
                    "attempt": attempt,
                    "status": Status.SUCCEEDED.value,
                    "reused": True,
                    "dependencies": completed.get("config_dependencies"),
                    # Reuse consumes no wall time; mark it as a
                    # zero-length point so it still lands on the axis.
                    "start_seconds": timestamp,
                    "end_seconds": timestamp,
                }
            )
            self._complete_stage(run_id, stage_index)
        self._note(
            "reuse",
            run_id=run_id,
            stage_index=stage_index,
            stage_attempt=attempt,
            detail="等价配置命中前缀缓存",
        )
        return True

    def _materialize_reuses(self):
        reused = []
        while True:
            for run_id in list(self.batch.state.queue):
                stage_index = self.batch.state.stage_progress[run_id]
                if self._reuse_stage(run_id):
                    reused.append((run_id, stage_index))
                    break
            else:
                break
        if reused:
            self.batch._save()
            for run_id, stage_index in reused:
                self.prefixes.reclaim_after(run_id, stage_index)

    def _launch(self, device, memory, host, candidate):
        batch = self.batch
        run_id = candidate.run_id
        stage_index = candidate.stage_index
        host_reserve = candidate.host_reserve_kb
        gpu_reserve = candidate.gpu_reserve_kb
        with batch._state_lock:
            if run_id not in batch.state.queue:
                return False
            if not _admits(self.workers, host, host_reserve) or not _admits_gpu(
                self.workers, memory, device, gpu_reserve
            ):
                batch.state.queue.remove(run_id)
                batch.state.queue.appendleft(run_id)
                return False
            stage_attempt = batch.state.start(run_id, device, stage_index)
        experiment = batch.by_id[run_id]
        root = (
            experiment.output_dir
            / "attempts"
            / f"stage-{stage_index}"
            / str(stage_attempt)
        )
        process = None
        limit = None
        parent_socket = child_socket = None
        try:
            root.mkdir(parents=True, exist_ok=True)
            main_module = sys.modules.get("__main__")
            needs_main = any(
                cls.__module__ == "__main__"
                for cls in (*experiment.pipeline.stages, type(batch.store.serializer))
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
                    "settings": asdict(
                        replace(
                            settings,
                            retries=batch.max_retries,
                            coverage=batch.estimate_coverage,
                        )
                    ),
                    # What the framework configured for this attempt, for
                    # diagnosis; the shell keeps the last word.
                    "environment": dict(self._environment),
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
                    "serializer": batch.store.serializer,
                    "metrics_path": experiment._metrics.path,
                    "cache_root": experiment._cache_root,
                    "stage_index": stage_index,
                    "stage_attempt": stage_attempt,
                },
                PickleSerializer(),
            )
            # Persist this before process creation.  If the scheduler itself is
            # killed after the child starts, resume must advance the Stage attempt
            # so checkpoints and user retry logic see a genuine retry.
            batch._save()  # Durable active membership precedes process creation.
            self.prefixes.attempt_started(run_id)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = memory.uuid
            env["PYTHONPATH"] = os.pathsep.join(
                str(Path(path).resolve()) for path in sys.path
            )
            env.update(self._environment)
            # The unit is unique per attempt: systemd unloads a finished scope
            # asynchronously, and a retry must not collide with the previous one.
            unit = f"expman-{run_id[:12]}-{stage_index}-{stage_attempt}"
            limit = limits.memory_limit(unit, host_reserve)
            env["EXPMAN_CGROUP_UNIT"] = unit
            env["EXPMAN_HOST_LIMIT_KB"] = str(limit.cap_kb)
            if limit.cgroup is not None:
                env["EXPMAN_CGROUP"] = str(limit.cgroup)
            elif limit.mechanism == "none":
                # No cgroup of its own: the attempt inherits this process's, whose
                # counters cover the whole session. An empty value tells it to
                # report its own high-water mark instead.
                env["EXPMAN_CGROUP"] = ""
            else:
                # A systemd scope is not proof that a memory controller enforces
                # its limit. The worker resolves and verifies its own group.
                env.pop("EXPMAN_CGROUP", None)
            parent_socket, child_socket = socket.socketpair()
            parent_socket.setblocking(False)
            env["EXPMAN_IPC_FD"] = str(child_socket.fileno())
            # The driver accepts whole MiB. Record the limit sent to MPS.
            cap_mib = max(1, int(candidate.gpu_cap_kb // 1024))
            applied_gpu_kb = cap_mib * 1024
            env.update(self.mps.environment(memory.uuid, cap_mib))
            self._account()
            with (root / "output.log").open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    [
                        *limit.command,
                        sys.executable,
                        "-m",
                        "expman.scheduling._worker",
                        str(root),
                    ],
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                    pass_fds=(child_socket.fileno(),),
                )
            child_socket.close()
            child_socket = None
            if (
                limit.mechanism == "cgroup"
                and limit.cgroup is not None
                and not limits.join(limit.cgroup, process.pid)
            ):
                raise RuntimeError(
                    f"could not move worker {process.pid} into {limit.cgroup}"
                )
            now = perf_counter()
            self.workers[run_id] = Worker(
                process=process,
                experiment=experiment,
                device=device,
                uuid=memory.uuid,
                started=now,
                stage_index=stage_index,
                stage_attempt=stage_attempt,
                channel=Channel(parent_socket),
                host_reserve_kb=host_reserve,
                gpu_reserve_kb=gpu_reserve,
                gpu_cap_kb=applied_gpu_kb,
                expected=candidate.expected,
                limit=limit,
            )
            parent_socket = None
            self._note(
                "launch",
                run_id=run_id,
                stage_index=stage_index,
                stage_attempt=stage_attempt,
                device=device,
                detail=(
                    f"host {host_reserve / 1024:.0f}M gpu {gpu_reserve / 1024:.0f}M"
                    f" MPS 请求上限 {cap_mib}M"
                ),
            )
        except BaseException:
            if child_socket is not None:
                child_socket.close()
            if parent_socket is not None:
                parent_socket.close()
            if process is not None:
                _kill_group(process)
                process.wait()
                _close_stdin(process)
                self.workers.pop(run_id, None)
            with batch._state_lock:
                batch.state.rollback_start(run_id, stage_index)
            batch._save()
            raise
        return True

    def _finish(self, run_id, *, cancelled=None):
        worker = self.workers[run_id]
        self._account()
        finished = perf_counter()
        if cancelled and worker.process.poll() is None:
            worker.read_messages(self.batch.recorder)
            if worker.mps_client is not None:
                self.mps.terminate_client(*worker.mps_client)
        _kill_group(worker.process)  # Also reap experiment-owned helper processes.
        worker.process.wait()
        _close_stdin(worker.process)
        worker.read_messages(self.batch.recorder)
        duration = max(0.0, finished - worker.started)
        # The dashboard places each attempt on the Batch's effective clock,
        # which already excludes the gaps between run invocations.  The
        # attempt ends "now" on that clock, so its span needs no clock
        # alignment: start is simply the end minus the measured duration.
        span_end = self.batch.elapsed_seconds
        span_start = max(0.0, span_end - duration)
        capped, peak_kb = self._cap_report(worker)
        result = worker.finished or StageResult(
            run_id,
            worker.stage_index,
            worker.stage_attempt,
            Status.CANCELLED if cancelled else Status.FAILED,
            duration,
            cancelled or "WorkerExit",
            f"worker exited with code {worker.process.returncode}",
        )
        cuda_out_of_memory = devices.cuda_out_of_memory(
            result.error_type, result.error_message
        )
        gpu_capped = cuda_out_of_memory and result.status is not Status.SUCCEEDED
        if cancelled is not None or gpu_capped:
            capped = False
        host_retry_floor = None
        if capped and result.status is not Status.SUCCEEDED:
            # The limit, not the experiment, ended this attempt: it is retried with
            # a raised peak estimate instead of spending the failure budget, and it
            # carries the peak it reached as a lower bound for that estimate. An
            # attempt that finished its work inside a throttled limit still counts
            # as done; the raised estimate only affects the attempts after it.
            if worker.limit is not None and worker.limit.capped:
                # A kernel OOM can precede the worker's first usage sample.
                host_retry_floor = worker.host_limit_kb or worker.limit.cap_kb
            result = replace(
                result,
                status=Status.CANCELLED,
                error_type="MemoryLimit",
                error_message=(
                    "attempt stopped by its memory limit of "
                    f"{host_retry_floor / 1024:.0f} MiB"
                    if host_retry_floor is not None
                    else "the attempt's memory controller reported a refusal"
                ),
            )
        if gpu_capped:
            result = replace(
                result,
                status=Status.CANCELLED,
                error_type="GpuMemoryLimit",
                error_message=result.error_message,
            )
        with self.batch._state_lock:
            self.batch.state.stage_elapsed[run_id] += duration
            observation = {
                "run_id": run_id,
                "stage": worker.stage_index,
                "attempt": worker.stage_attempt,
                "device": worker.device,
                "uuid": worker.uuid,
                "concurrency": worker.area / max(duration, 1e-9),
                "duration": duration,
                "stage_duration_seconds": result.duration_seconds,
                "status": result.status.value,
                "peak_kb": peak_kb,
                "host_retry_floor_kb": host_retry_floor,
                # Highest NVIDIA process-memory sample for this worker PID.
                "gpu_process_peak_kb": worker.gpu_process_peak_kb,
                "gpu_retry_floor_kb": (
                    worker.gpu_cap_kb or settings.gpu_peak_kb_default
                )
                if gpu_capped
                else None,
                "gpu_limit_kb": worker.gpu_cap_kb,
                "gpu_limit_verified": worker.gpu_limit_verified,
                "gpu_capped": gpu_capped,
                "capped": capped,
                "reused": worker.reused,
                "dependencies": worker.dependencies,
                "start_seconds": round(span_start, 2),
                "end_seconds": round(span_end, 2),
            }
            self.batch.state.stage_history.append(observation)
            self.stages.invalidate(worker.stage_index)
            self.batch.state.release(run_id)
            self.batch.live.running_info.pop(run_id, None)
            del self.workers[run_id]
            if worker.channel is not None:
                worker.channel.socket.close()
            if result.status is Status.SUCCEEDED:
                self._complete_stage(run_id, worker.stage_index, observation)
            else:
                self.batch.record_attempt(
                    run_id,
                    result.status,
                    error_type=result.error_type,
                    error_message=result.error_message,
                )
                if (
                    result.status is Status.CANCELLED
                    or self.batch.state.failed_attempts(run_id, worker.stage_index)
                    <= self.batch.max_retries
                ):
                    self.batch.state.retry(run_id)
            self._note(
                "finish",
                run_id=run_id,
                stage_index=worker.stage_index,
                stage_attempt=worker.stage_attempt,
                device=worker.device,
                status=str(result.status),
                detail=f"{duration:.1f}s 峰值 {peak_kb / 1024:.0f}M",
            )
            self.batch._save()

        self.prefixes.reclaim_after(
            run_id, worker.stage_index, succeeded=result.status is Status.SUCCEEDED
        )

    def run(self):
        try:
            self.prefixes.reclaim_saved()
            while self.batch.state.queue or self.workers:
                for run_id, worker in list(self.workers.items()):
                    worker.read_messages(self.batch.recorder)
                    if worker.process.poll() is not None:
                        self._finish(run_id)
                if not self.batch.state.queue and not self.workers:
                    break
                self._account()
                # Both readings are rechecked every tick: a stale ratio never
                # admits work, and an unreadable one keeps every card closed.
                memory, host = self._observe()
                self._relieve(memory, host)
                if memory is not None and devices.fits_reserve(host, 0):
                    self._materialize_reuses()
                    if self.batch.state.queue and self.mps.ready(
                        item.uuid for item in memory.values()
                    ):
                        self._launch_available(memory, host)
                sleep(settings.poll_interval)
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
        finally:
            self.mps.close()
