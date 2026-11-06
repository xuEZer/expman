"""Memory-gated per-GPU process scheduling and durable attempt collection."""

import math
import os
import signal
import socket
import subprocess
import sys
import warnings
from collections import deque
from dataclasses import replace
from pathlib import Path
from time import perf_counter, sleep, time

import expman.config.devices as devices
import expman.scheduling.environment as environment
import expman.scheduling.limits as limits
from expman.core.dependencies import declared_paths
from expman.core.dependencies import observation as config_observation
from expman.core.events import ProgressEvent, Status
from expman.core.experiment import AttemptResult, SnapshotOutput, StageResult
from expman.estimation.duration_model import DurationModel
from expman.estimation.memory_model import LocalQuantileEstimator, PeakCeiling
from expman.estimation.time_model import encode
from expman.scheduling._models import (
    Candidate,
    HostGate,
    Plan,
    Worker,
    _close_stdin,
    _kill_group,
)
from expman.scheduling._mps import MpsService
from expman.scheduling._planning import (
    PLAN_BEAM_WIDTH,
    PLAN_CANDIDATES_PER_STAGE,
    _interleave,
    _plan_score,
)
from expman.scheduling._resource import (
    _admits,
    _admits_gpu,
    _device_unbacked,
    _resource_left,
)
from expman.scheduling.ipc import Channel
from expman.storage.base import PickleSerializer, RecoveryWarning, write_record


class GpuScheduler:
    def __init__(self, batch):
        if os.name != "posix":
            raise RuntimeError("GPU process scheduling currently requires Linux or WSL")
        self.batch = batch
        self.monitor = devices.NvidiaMemory(batch.devices)
        self.host = devices.MeminfoMonitor()
        self.host_gate = HostGate()
        # Resolved once per run: a bad EXPMAN_THREADS fails here, before any
        # attempt has been started or any output written.
        self._environment = environment.overlay()
        self.mps = MpsService()
        self.workers = {}
        self.last_tick = perf_counter()
        # Peak host memory per attempt: admission charges the configuration-based
        # estimate and the cgroup cap enforces it, so the scheduler keeps no
        # per-run table of its own here.
        self.peaks = PeakCeiling.from_history(batch._gpu_history)
        self._stage_peak_models = {}
        self._stage_duration_models = {}
        # Progress rendering and ETA calculation ask for Stage groups often.
        # Dependencies are declared by each Stage, so the identity of a run is
        # known before it executes and never changes; cache it per run and Stage.
        self._declared_paths_cache = {}
        self._prefix_paths_cache = {}
        self._stage_group_cache = {}
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

    def _stage_label(self, stage_index):
        """The Stage class name for display; any experiment defines the order."""
        experiments = self.batch.experiments
        if not experiments:
            return None
        stages = getattr(experiments[0].pipeline, "stages", ())
        if 0 <= stage_index < len(stages):
            return getattr(stages[stage_index], "__name__", None)
        return None

    def _declared_paths(self, run_id, stage_index):
        """Return the paths one Stage declares for one run."""
        cached = self._declared_paths_cache.setdefault(run_id, {})
        paths = cached.get(stage_index)
        if paths is None:
            experiment = next(
                item for item in self.batch.experiments if item.run_id == run_id
            )
            stage_type = experiment.pipeline.stages[stage_index]
            paths = declared_paths(stage_type.config_dependencies(experiment._frozen))
            cached[stage_index] = paths
        return paths

    def _prefix_paths(self, run_id, stage_index):
        """Return the rolling declared dependencies of one Stage's inputs.

        A Stage receives the output of every preceding Stage, so their declared
        dependencies are part of its input identity as well: when they match, the
        upstream value is reusable.
        """
        cached = self._prefix_paths_cache.setdefault(run_id, {})
        paths = cached.get(stage_index)
        if paths is None:
            collected = set()
            for index in range(stage_index + 1):
                collected.update(self._declared_paths(run_id, index))
            paths = frozenset(collected)
            cached[stage_index] = paths
        return paths

    def _append_stage_history(self, entry):
        """Record one Stage result for diagnostics and resource sampling."""
        self.batch._stage_history.append(entry)

    def _stage_group(self, run_id, stage_index):
        """Return the reusable work identity of one top-level Stage."""
        cached = self._stage_group_cache.setdefault(stage_index, {})
        if run_id in cached:
            return cached[run_id]
        experiment = next(
            item for item in self.batch.experiments if item.run_id == run_id
        )
        config = experiment._frozen
        paths = sorted(self._prefix_paths(run_id, stage_index), key=repr)
        group = (
            stage_index,
            config_observation(config, ("seed",)),
            tuple((path, config_observation(config, path)) for path in paths),
        )
        cached[run_id] = group
        return group

    def _remaining_stage_groups(self, queue, running):
        """Return one representative for every not-yet-materialized Stage group."""
        completed = {
            self._stage_group(entry["run_id"], entry["stage"])
            for entry in self.batch._stage_history
            if (
                entry.get("status") == Status.SUCCEEDED.value
                and isinstance(entry.get("run_id"), str)
                and isinstance(entry.get("stage"), int)
            )
        }
        active = {}
        for run_id, details in running.items():
            stage_index = details.get("stage_index")
            if not isinstance(stage_index, int):
                return None
            active[self._stage_group(run_id, stage_index)] = (run_id, stage_index)
        groups = dict(active)
        # A running attempt still has downstream work. Keep its current group
        # active, then include the stages that will become ready after it ends.
        starts = {run_id: self.batch._stage_progress[run_id] for run_id in queue}
        starts.update(
            (run_id, details["stage_index"] + 1) for run_id, details in running.items()
        )
        for run_id, start in starts.items():
            experiment = next(
                item for item in self.batch.experiments if item.run_id == run_id
            )
            for stage_index in range(start, len(experiment.pipeline.stages)):
                group = self._stage_group(run_id, stage_index)
                if group not in completed and group not in groups:
                    groups[group] = (run_id, stage_index)
        return groups, set(active)

    def stage_completion(self):
        """Return completed and total reusable parameter groups per Stage."""
        totals = {}
        for experiment in self.batch.experiments:
            for stage_index in range(len(experiment.pipeline.stages)):
                totals.setdefault(stage_index, set()).add(
                    self._stage_group(experiment.run_id, stage_index)
                )
        completed = {stage_index: set() for stage_index in totals}
        for entry in self.batch._stage_history:
            if entry.get("status") != Status.SUCCEEDED.value:
                continue
            run_id, stage_index = entry.get("run_id"), entry.get("stage")
            if not isinstance(run_id, str) or stage_index not in completed:
                continue
            completed[stage_index].add(self._stage_group(run_id, stage_index))
        return tuple(
            (stage_index, len(completed[stage_index]), len(groups))
            for stage_index, groups in sorted(totals.items())
        )

    def _stage_vectors(self, stage_index):
        """Encode each run's declared dependency prefix for a Stage input."""
        configs = []
        for experiment in self.batch.experiments:
            cfg = experiment._frozen
            projected = {}
            for path in self._prefix_paths(experiment.run_id, stage_index):
                value = cfg
                try:
                    for key in path:
                        value = value[key]
                except (KeyError, IndexError, TypeError):
                    value = None
                projected[repr(path)] = value
            configs.append(projected)
        return encode(configs), {
            experiment.run_id: index
            for index, experiment in enumerate(self.batch.experiments)
        }

    def _stage_peak(self, run_id, stage_index, field, default):
        """Lazily fit one configuration-based peak model per top-level Stage."""
        key = (stage_index, field)
        model = self._stage_peak_models.get(key)
        if model is None:
            encoding, indices = self._stage_vectors(stage_index)
            model = LocalQuantileEstimator(
                encoding,
                indices,
                default,
                include_zero=False,
            )
            for entry in self.batch._stage_history:
                if entry.get("stage") != stage_index or entry.get("reused"):
                    continue
                value = entry.get(field)
                if field == "gpu_process_peak_kb":
                    if entry.get("gpu_capped"):
                        value = max(
                            value or 0.0, entry.get("gpu_retry_floor_kb") or 0.0
                        )
                        if not value:
                            continue
                    elif entry.get("status") != Status.SUCCEEDED.value:
                        # Shedding can interrupt a process before it initializes CUDA.
                        continue
                model.record(
                    entry.get("run_id"),
                    value,
                    capped=entry.get(
                        "capped" if field == "peak_kb" else "gpu_capped", False
                    ),
                )
            self._stage_peak_models[key] = model
        return model.estimate(run_id)

    def _stage_duration(self, run_id, stage_index):
        """Estimate one Stage duration interval from its dependency samples.

        Models are cached and invalidated only when a Stage result arrives.  Tick
        processing merely reads the latest model; it never treats elapsed running
        time as a new sample.  A cancelled attempt measured only a lower bound, so
        it widens the lower side of the interval and stays out of the center and
        the upper side, where it would make the answer too tight.
        """
        model = self._stage_duration_models.get(stage_index)
        if model is None:
            encoding, indices = self._stage_vectors(stage_index)
            model = DurationModel(
                encoding, indices, coverage=self.batch.estimate_coverage
            )
            measured = False
            for entry in self.batch._stage_history:
                if entry.get("stage") != stage_index or entry.get("reused"):
                    continue
                duration = entry.get("stage_duration_seconds")
                if not isinstance(duration, (int, float)) or duration <= 0:
                    continue
                status = entry.get("status")
                if status == Status.SUCCEEDED.value:
                    measured = True
                    model.record(entry.get("run_id"), duration)
                elif status == Status.CANCELLED.value:
                    model.record(entry.get("run_id"), duration, censored=True)
            if not measured:
                self._stage_duration_models[stage_index] = False
                return None
            self._stage_duration_models[stage_index] = model
        if model is False:
            return None
        return model.predict(run_id)

    def _stage_candidates(
        self,
        stage_index,
        gpu_capacity_kb,
        host_probe_kb: float = devices.HOST_PEAK_KB_DEFAULT,
        gpu_probe_kb: float = devices.GPU_PEAK_KB_DEFAULT,
    ):
        """Return diverse ready candidates for one Stage and its own history only."""
        batch = self.batch
        # A group with an attempt already running must not start another member.
        # The running member has left the queue and has not published its result
        # yet, so without this the next queued member becomes the group's
        # representative on the following tick -- and the tick after that the one
        # after it.  A Stage that runs for longer than a tick then launches its
        # whole group before the first result exists, which is the duplicate work
        # the grouping exists to avoid.  One member per group runs at a time; the
        # rest reuse its result.
        running = {
            self._stage_group(run_id, stage_index)
            for run_id, worker in self.workers.items()
            if worker.stage_index == stage_index
        }
        run_ids = [
            run_id
            for run_id in batch._queue
            if batch._stage_progress[run_id] == stage_index
            and self._stage_group(run_id, stage_index) not in running
        ]
        # Declared dependency groups are known before execution. Keep exactly one
        # representative runnable; cache materialization advances every other
        # member once that representative finishes.
        representatives = {}
        for run_id in run_ids:
            group = self._stage_group(run_id, stage_index)
            previous = representatives.get(group)
            if previous is None or (
                f"{run_id}:{stage_index}" in batch._stage_attempts
                and f"{previous}:{stage_index}" not in batch._stage_attempts
            ):
                representatives[group] = run_id
        run_ids = list(representatives.values())
        if not run_ids:
            return []
        encoding, indices = self._stage_vectors(stage_index)
        rows = encoding.rows
        sampled = [
            indices[entry["run_id"]]
            for entry in batch._stage_history
            if (
                entry.get("stage") == stage_index
                and not entry.get("reused")
                and entry.get("run_id") in indices
            )
        ]

        def novelty(run_id):
            if not sampled:
                return 1.0
            row = rows[indices[run_id]][1:]
            return min(
                sum(
                    (left - right) ** 2
                    for left, right in zip(row, rows[item][1:], strict=True)
                )
                for item in sampled
            )

        retries, fresh = [], []
        for run_id in run_ids:
            value = novelty(run_id)
            target = (
                retries if f"{run_id}:{stage_index}" in batch._stage_attempts else fresh
            )
            target.append((run_id, value))
        # Retries remain first within their own Stage.  New combinations then
        # spread samples across that Stage's dependency-feature space.
        ordered = retries + sorted(fresh, key=lambda item: item[1], reverse=True)
        candidates = []
        selected = ordered[:PLAN_CANDIDATES_PER_STAGE]
        denominator = max(1, len(selected) - 1)
        for rank, (run_id, _value) in enumerate(selected):
            host_estimate = self._stage_peak(
                run_id, stage_index, "peak_kb", host_probe_kb
            )
            gpu_estimate = self._stage_peak(
                run_id, stage_index, "gpu_process_peak_kb", gpu_probe_kb
            )
            candidates.append(
                Candidate(
                    run_id,
                    stage_index,
                    host_estimate,
                    min(gpu_estimate, gpu_capacity_kb),
                    min(limits.cap_kb(gpu_estimate), gpu_capacity_kb),
                    self._stage_duration(run_id, stage_index),
                    1.0 - rank / denominator,
                )
            )
        return candidates

    def _account(self):
        now = perf_counter()
        for worker in self.workers.values():
            count = sum(item.device == worker.device for item in self.workers.values())
            worker.area += (now - max(self.last_tick, worker.started)) * count
            if limits.refused(worker.memory_events):
                worker.cap_hit = True
        self.last_tick = now
        with self.batch._state_lock:
            self.batch._gpu_running_info = {
                run_id: {
                    "device": worker.device,
                    "uuid": worker.uuid,
                    "concurrency": worker.area / max(now - worker.started, 1e-9),
                    "duration": max(0.0, now - worker.started),
                    "expected_seconds": None
                    if worker.expected is None
                    else worker.expected.center,
                    "expected_lower_seconds": None
                    if worker.expected is None
                    else worker.expected.lower,
                    "expected_upper_seconds": None
                    if worker.expected is None
                    else worker.expected.upper,
                    "stage_index": worker.stage_index,
                    "resident_kb": worker.resident_kb,
                    "peak_kb": worker.peak_kb,
                    "gpu_used_kb": worker.gpu_used_kb,
                    "gpu_process_peak_kb": worker.gpu_process_peak_kb,
                }
                for run_id, worker in self.workers.items()
            }

    def _messages(self, worker):
        """Consume worker IPC without querying its process or cgroup."""
        if worker.channel is None:
            return
        for message in worker.channel.drain():
            if not isinstance(message, dict):
                continue
            kind = message.get("type")
            if kind == "event":
                event = message["event"]
                if (
                    isinstance(event, ProgressEvent)
                    and event.run_id == worker.experiment.run_id
                    and event.attempt == worker.stage_attempt
                ):
                    # The IPC channel identifies this attempt even when a nested
                    # Pipeline uses a child-local stage_id. Preserve the event.
                    worker.progress = event
                self.batch.recorder.record(event)
            elif kind == "mps_ready":
                worker.mps_client = (message["server"], message["pid"])
            elif kind == "startup_failed":
                worker.finished = StageResult(
                    worker.experiment.run_id,
                    worker.stage_index,
                    worker.stage_attempt,
                    Status.FAILED,
                    0.0,
                    message["error_type"],
                    message["error_message"],
                )
            elif kind == "resource":
                self._resources(worker, message)
            elif kind == "finished" and isinstance(message.get("result"), StageResult):
                worker.finished = message["result"]
                dependencies = message.get("dependencies")
                worker.dependencies = (
                    dependencies if isinstance(dependencies, list) else None
                )
                worker.reused = message.get("reused") is True
                resources = message.get("resources")
                if isinstance(resources, dict):
                    self._resources(worker, resources)

    @staticmethod
    def _resources(worker, message):
        if "gpu_used_kb" in message:
            used = message["gpu_used_kb"]
            if isinstance(used, (int, float)) and math.isfinite(used) and used >= 0:
                worker.gpu_used_kb = used
                worker.gpu_process_peak_kb = max(worker.gpu_process_peak_kb or 0, used)
            else:
                worker.gpu_used_kb = None
        current, peak = message.get("host_current_kb"), message.get("host_peak_kb")
        if all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
            for value in (current, peak)
        ):
            worker.resident_kb = current
            worker.peak_kb = max(worker.peak_kb, current, peak)
        for name in ("host_usage_source", "host_usage_error"):
            value = message.get(name)
            if value is None or isinstance(value, str):
                setattr(worker, name, value)
        if "host_limit_verified" in message and worker.limit is not None:
            observed_cap = message.get("host_limit_kb")
            valid_cap = (
                isinstance(observed_cap, (int, float))
                and not isinstance(observed_cap, bool)
                and math.isfinite(observed_cap)
                and observed_cap >= 0
            )
            worker.host_limit_kb = observed_cap if valid_cap else None
            verified = (
                message["host_limit_verified"] is True
                and valid_cap
                and limits.cap_matches_request(observed_cap, worker.limit.cap_kb)
            )
            # cap_kb retains the requested ceiling, even if this observation
            # cannot verify an effective per-attempt kernel limit.
            worker.limit = replace(worker.limit, verified=verified)
        events = message.get("memory_events")
        if isinstance(events, dict):
            worker.memory_events = events
        directory = message.get("cgroup")
        if worker.limit is not None and "cgroup" in message:
            worker.limit = replace(
                worker.limit,
                cgroup=Path(directory) if isinstance(directory, str) else None,
            )

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
                self.batch._host_memory = {}
                self.batch._device_memory = {}
            return None, None
        # One more attempt is charged at least the largest peak seen so far.
        admits_next = _admits(
            self.workers, host, max(self.peaks.ceiling_kb, devices.HOST_PEAK_KB_DEFAULT)
        )
        with self.batch._state_lock:
            self.batch._host_memory = {
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
            self.batch._device_memory = {
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
        shortage with one worker therefore pauses launches (see ``HostGate``
        and ``_admits_gpu``) and keeps the attempt.

        The host check is admission's own arithmetic with nothing new to place:
        if one more attempt of zero size would not fit, what the host cannot
        honour is the reservations already running. Pressure that is not the
        host's is looked for on each card in turn. One attempt per shedding,
        and none while a previous one is still being released.
        """
        if not self.workers or perf_counter() < self._relief_until:
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
        if len(self.workers) < 2:
            return
        # The attempt launched last is the one whose reservation this machine
        # has had least time to honour, and the cheapest to give back.
        self._shed(self._newest(device), kind, device)
        self._relief_until = perf_counter() + devices.RELIEF_COOLDOWN

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
            capped = limits.near_cap(peak, worker.limit.cap_kb) or (
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
        if host_capacity <= 0 or not devices_in_plan:
            self.last_plan = {
                "at": time(),
                "ready": None,
                "host_fits": False if host_capacity <= 0 else None,
                "gpu_fits": None,
                "capacity": {
                    "host_left_kb": host_capacity,
                    "gpu_left_kb": available_gpu,
                },
                "placements": [],
                "launched": [],
            }
            return ()
        gpu_capacities = tuple(
            max(0.0, available_gpu[device]) for device in devices_in_plan
        )
        # Admission uses current headroom; scoring uses fixed card totals so
        # work already occupying a card remains visible across scheduling ticks.
        gpu_totals = tuple(memory[device].total * 1024 for device in devices_in_plan)
        stages = sorted(
            {self.batch._stage_progress[run_id] for run_id in self.batch._queue}
        )
        gpu_capacity_kb = min(memory[device].total * 1024 for device in devices_in_plan)
        # An unmeasured Stage probes at a bounded share of the device instead of
        # running uncapped one at a time: the probe ceiling is what the candidate
        # is priced at, so the number of concurrent probes falls out of the
        # packing arithmetic rather than out of a serialisation rule.
        host_probe_kb = devices.probe_cap_kb(
            host.total_kb, devices.HOST_PEAK_KB_DEFAULT, self.peaks.ceiling_kb
        )
        gpu_probe_kb = devices.probe_cap_kb(
            gpu_capacity_kb, devices.GPU_PEAK_KB_DEFAULT
        )
        candidates = _interleave(
            [
                self._stage_candidates(
                    stage, gpu_capacity_kb, host_probe_kb, gpu_probe_kb
                )
                for stage in stages
            ]
        )
        plans = [Plan(host_capacity, gpu_capacities)]
        for candidate in candidates:
            expanded = list(plans)
            for plan in plans:
                if candidate.host_reserve_kb > plan.host_left_kb:
                    continue
                for index, gpu_left in enumerate(plan.gpu_left_kb):
                    if candidate.gpu_reserve_kb > gpu_left:
                        continue
                    left = list(plan.gpu_left_kb)
                    left[index] -= candidate.gpu_reserve_kb
                    expanded.append(
                        Plan(
                            plan.host_left_kb - candidate.host_reserve_kb,
                            tuple(left),
                            (*plan.placements, (candidate, devices_in_plan[index])),
                            plan.novelty + candidate.novelty,
                        )
                    )
            plans = sorted(
                expanded,
                key=lambda plan: _plan_score(plan, host_capacity, gpu_totals),
                reverse=True,
            )[:PLAN_BEAM_WIDTH]
        chosen = max(
            plans,
            key=lambda plan: _plan_score(plan, host_capacity, gpu_totals),
        )
        # Published for the dashboard: what this tick could place and what it
        # passed over.  Recorded here because only this thread may read the
        # estimators that produced it.
        self.last_plan = {
            "at": time(),
            "ready": len(candidates),
            "host_fits": any(c.host_reserve_kb <= host_capacity for c in candidates)
            if candidates
            else None,
            "gpu_fits": any(c.gpu_reserve_kb <= max(gpu_capacities) for c in candidates)
            if candidates
            else None,
            "capacity": {
                "host_left_kb": host_capacity,
                "gpu_left_kb": dict(available_gpu),
            },
            "placements": [
                {
                    "run_id": candidate.run_id,
                    "stage_index": candidate.stage_index,
                    "stage": self._stage_label(candidate.stage_index),
                    "device": device,
                    "host_reserve_kb": candidate.host_reserve_kb,
                    "gpu_reserve_kb": candidate.gpu_reserve_kb,
                    "gpu_cap_kb": candidate.gpu_cap_kb,
                }
                for candidate, device in chosen.placements
            ],
            "launched": [],
        }
        return chosen.placements

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

    def _materialize_reuses(self):
        """Advance cache-equivalent Stage members without creating a worker."""
        changed = False
        while True:
            advanced = False
            for run_id in list(self.batch._queue):
                experiment = next(
                    item for item in self.batch.experiments if item.run_id == run_id
                )
                stage_index = self.batch._stage_progress[run_id]
                store = experiment._store
                shared = store.shared
                if shared is None:
                    continue
                parent = "root"
                if stage_index:
                    parent_ref = store.completed_reference((stage_index - 1,))
                    if parent_ref is None:
                        continue
                    parent = parent_ref[2]
                # Metadata only: a reusable Stage's output can be large, and
                # reading it here would cost that much memory in the scheduler
                # for every member that reuses it.  The worker that consumes the
                # output loads it once, when it needs it.
                candidate = shared.find_metadata(
                    parent, (stage_index,), experiment._frozen
                )
                if candidate is None:
                    continue
                reference, completed = candidate
                attempt = (
                    self.batch._stage_attempts.get(f"{run_id}:{stage_index}", 0) + 1
                )
                store.metrics.restore(run_id, attempt, completed.get("metrics", []))
                store.reference((stage_index,), reference, reused=True)
                store.status(
                    (stage_index,),
                    Status.SUCCEEDED.value,
                    attempt,
                    reused=True,
                    elapsed_seconds=completed.get("elapsed_seconds"),
                    restore_seconds=0.0,
                )
                self.batch._stage_attempts[f"{run_id}:{stage_index}"] = attempt
                self.batch._queue.remove(run_id)
                self._append_stage_history(
                    {
                        "run_id": run_id,
                        "stage": stage_index,
                        "attempt": attempt,
                        "status": Status.SUCCEEDED.value,
                        "reused": True,
                        "dependencies": completed.get("config_dependencies"),
                        # Reuse consumes no wall time; mark it as a
                        # zero-length point so it still lands on the axis.
                        "start_seconds": round(self.batch.elapsed_seconds, 2),
                        "end_seconds": round(self.batch.elapsed_seconds, 2),
                    }
                )
                if stage_index == len(experiment.pipeline.stages) - 1:
                    experiment._attempts.append(
                        AttemptResult(
                            run_id,
                            len(experiment.result.attempts) + 1,
                            Status.SUCCEEDED,
                            self.batch._stage_elapsed[run_id],
                            output_source=SnapshotOutput(store, (stage_index,)),
                        )
                    )
                else:
                    self.batch._stage_progress[run_id] += 1
                    self.batch._queue.append(run_id)
                changed = advanced = True
                self._note(
                    "reuse",
                    run_id=run_id,
                    stage_index=stage_index,
                    stage_attempt=attempt,
                    detail="等价配置命中前缀缓存",
                )
                break
            if not advanced:
                break
        if changed:
            self.batch._save()
        return changed

    def _launch(self, device, memory, host, candidate):
        batch = self.batch
        with batch._state_lock:
            if candidate.run_id not in batch._queue:
                return False
            batch._queue.remove(candidate.run_id)
        run_id = candidate.run_id
        stage_index = candidate.stage_index
        host_reserve = candidate.host_reserve_kb
        gpu_reserve = candidate.gpu_reserve_kb
        duration_estimate = candidate.expected
        gpu_fits = _admits_gpu(self.workers, memory, device, gpu_reserve)
        if not _admits(self.workers, host, host_reserve) or not gpu_fits:
            # Put the selection back at the front. Dispatching updated the
            # estimator's own state, and re-selecting the same run is idempotent,
            # so the queue keeps its remaining order while the host is short.
            with batch._state_lock:
                batch._queue.appendleft(run_id)
            return False
        with batch._state_lock:
            batch._active_gpu[run_id] = device
        experiment = next(item for item in batch.experiments if item.run_id == run_id)
        stage_key = f"{run_id}:{stage_index}"
        previous_stage_attempt = batch._stage_attempts.get(stage_key)
        stage_attempt = (previous_stage_attempt or 0) + 1
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
                    "serializer": batch._serializer,
                    "metrics_path": experiment._metrics.path,
                    "cache_root": experiment._cache_root,
                    "stage_index": stage_index,
                    "stage_attempt": stage_attempt,
                    "attempts": [
                        replace(item, output_source=None)
                        for item in experiment.result.attempts
                    ],
                },
                PickleSerializer(),
            )
            # Persist this before process creation.  If the scheduler itself is
            # killed after the child starts, resume must advance the Stage attempt
            # so checkpoints and user retry logic see a genuine retry.
            batch._stage_attempts[stage_key] = stage_attempt
            batch._save()  # Durable active membership precedes process creation.
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
            env["EXPMAN_POLL_INTERVAL"] = str(devices.POLL_INTERVAL)
            # An unmeasured Stage probes at the share priced into its candidate;
            # a measured Stage is capped at its estimate plus room to grow. Both
            # feed whatever they reach back to the estimator either way.
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
                root=root,
                started=now,
                stage_index=stage_index,
                stage_attempt=stage_attempt,
                channel=Channel(parent_socket),
                host_reserve_kb=host_reserve,
                gpu_reserve_kb=gpu_reserve,
                gpu_cap_kb=applied_gpu_kb,
                expected=duration_estimate,
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
                    + (
                        f" MPS 请求上限 {applied_gpu_kb / 1024:.0f}M"
                        if applied_gpu_kb > 0
                        else ""
                    )
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
                batch._active_gpu.pop(run_id, None)
                if previous_stage_attempt is None:
                    batch._stage_attempts.pop(stage_key, None)
                else:
                    batch._stage_attempts[stage_key] = previous_stage_attempt
                batch._queue.appendleft(run_id)
            batch._save()
            raise
        return True

    def _finish(self, run_id, *, cancelled=None):
        worker = self.workers[run_id]
        self._account()
        finished = perf_counter()
        if cancelled and worker.process.poll() is None:
            self._messages(worker)
            if worker.mps_client is not None:
                self.mps.terminate_client(*worker.mps_client)
        _kill_group(worker.process)  # Also reap experiment-owned helper processes.
        worker.process.wait()
        _close_stdin(worker.process)
        self._messages(worker)
        experiment = worker.experiment
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
        if capped and result.status is not Status.SUCCEEDED:
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
                    if worker.limit is not None and worker.limit.capped
                    else "attempt was killed while running without a memory limit"
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
            stage_key = f"{run_id}:{worker.stage_index}"
            self.batch._stage_attempts[stage_key] = worker.stage_attempt
            self.batch._stage_elapsed[run_id] += duration
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
                # Old allocator peaks have a different scope. A distinct key
                # lets resumed batches retain history without fitting to it.
                "gpu_process_peak_kb": worker.gpu_process_peak_kb,
                "gpu_retry_floor_kb": (worker.gpu_cap_kb or devices.GPU_PEAK_KB_DEFAULT)
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
            self._append_stage_history(observation)
            self._stage_peak_models.pop((worker.stage_index, "peak_kb"), None)
            self._stage_peak_models.pop(
                (worker.stage_index, "gpu_process_peak_kb"), None
            )
            self._stage_duration_models.pop(worker.stage_index, None)
            self.batch._active_gpu.pop(run_id)
            self.batch._gpu_running_info.pop(run_id, None)
            del self.workers[run_id]
            if worker.channel is not None:
                worker.channel.socket.close()
            final_stage = worker.stage_index == len(experiment.pipeline.stages) - 1
            if result.status is Status.SUCCEEDED and not final_stage:
                self.batch._stage_progress[run_id] += 1
                self.batch._queue.append(run_id)
            elif result.status is Status.SUCCEEDED:
                final = AttemptResult(
                    run_id,
                    len(experiment.result.attempts) + 1,
                    Status.SUCCEEDED,
                    self.batch._stage_elapsed[run_id],
                    output_source=SnapshotOutput(
                        experiment._store, (worker.stage_index,)
                    ),
                )
                experiment._attempts.append(final)
                self.batch._gpu_history.append(observation)
                if not worker.reused:
                    self.peaks.record(observation)
            elif (
                result.status is Status.CANCELLED
                or worker.stage_attempt <= self.batch.max_retries
            ):
                experiment._attempts.append(
                    AttemptResult(
                        run_id,
                        len(experiment.result.attempts) + 1,
                        result.status,
                        self.batch._stage_elapsed[run_id],
                        error_type=result.error_type,
                        error_message=result.error_message,
                    )
                )
                self.batch._queue.append(run_id)
            else:
                experiment._attempts.append(
                    AttemptResult(
                        run_id,
                        len(experiment.result.attempts) + 1,
                        Status.FAILED,
                        self.batch._stage_elapsed[run_id],
                        error_type=result.error_type,
                        error_message=result.error_message,
                    )
                )
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

    def run(self):
        try:
            while self.batch._queue or self.workers:
                for run_id, worker in list(self.workers.items()):
                    self._messages(worker)
                    if worker.process.poll() is not None:
                        self._finish(run_id)
                if not self.batch._queue and not self.workers:
                    break
                self._account()
                # Both readings are rechecked every tick: a stale ratio never
                # admits work, and an unreadable one keeps every card closed.
                memory, host = self._observe()
                self._relieve(memory, host)
                launching = self.host_gate.can_launch(host)
                if launching and memory is not None:
                    self._materialize_reuses()
                    if self.batch._queue and self.mps.ready(
                        item.uuid for item in memory.values()
                    ):
                        self._launch_available(memory, host)
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
        finally:
            self.mps.close()
