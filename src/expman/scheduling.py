"""Memory-gated per-GPU process scheduling and durable attempt collection."""

import os
import signal
import socket
import subprocess
import sys
import warnings
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter, sleep

from . import devices, limits
from ._memory_model import LocalQuantileEstimator, PeakEstimator
from ._time_model import features
from .dependencies import observation as config_observation
from .events import Status
from .experiment import AttemptResult, SnapshotOutput, StageResult
from .ipc import Channel
from .storage import PickleSerializer, RecoveryWarning, write_record

PLAN_CANDIDATES_PER_STAGE = 32
PLAN_BEAM_WIDTH = 128


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
    device: int | None
    uuid: str | None
    root: Path
    started: float
    stage_index: int | None = 0
    stage_attempt: int = 1
    channel: Channel | None = None
    gpu_contract_kb: float = 0.0
    expected_seconds: float | None = None
    area: float = 0.0
    offset: int = 0
    limit: object = None
    # Sampled every tick: resident is what the worker holds now, peak is the most
    # it has ever held. Admission reserves the gap between the two.
    resident_kb: float = 0.0
    peak_kb: float = 0.0
    gpu_allocated_kb: float | None = None
    gpu_reserved_kb: float | None = None
    gpu_peak_allocated_kb: float | None = None
    gpu_peak_reserved_kb: float | None = None
    memory_events: dict = field(default_factory=dict)
    finished: StageResult | None = None
    dependencies: list | None = None
    stage_reports: list | None = None
    reused: bool = False
    probe: bool = False
    cap_hit: bool = False


@dataclass(frozen=True)
class Candidate:
    """One ready Stage with its resource contracts for this scheduling tick."""

    run_id: str
    stage_index: int
    host_estimate_kb: float
    host_contract_kb: float
    gpu_contract_kb: float
    expected_seconds: float | None
    novelty: float


@dataclass(frozen=True)
class Plan:
    """A feasible placement of ready Stage candidates on the selected cards."""

    host_left_kb: float
    gpu_left_kb: tuple[float, ...]
    placements: tuple[tuple[Candidate, int], ...] = ()
    novelty: float = 0.0


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
        self.host_gate = HostGate()
        self.workers = {}
        self.last_tick = perf_counter()
        # Peak host memory per attempt: admission charges the configuration-based
        # estimate and the cgroup cap enforces it, so the scheduler keeps no
        # per-run table of its own here.
        self.peaks = PeakEstimator.from_batch(batch)
        self._stage_peak_models = {}
        self._stage_duration_models = {}
        batch._scheduler = self
        self._probe_pending = not any(
            experiment.result.attempts for experiment in batch.experiments
        )

    def _stage_paths(self, stage_index):
        """Return the rolling configuration dependencies for one top-level Stage.

        A Stage receives the output of every preceding Stage.  Their reads are
        therefore part of its input identity as well: when those reads match,
        the upstream value is reusable.  Current-Stage reads join that prefix as
        soon as the first sample supplies them.
        """
        paths = set()
        for entry in self.batch._stage_history:
            if entry.get("stage", -1) > stage_index:
                continue
            for dependency in entry.get("dependencies") or ():
                if isinstance(dependency, tuple) and len(dependency) == 2:
                    path, _digest = dependency
                    if isinstance(path, tuple):
                        paths.add(path)
        return paths

    def _stage_group(self, run_id, stage_index):
        """Return the reusable work identity of one top-level Stage."""
        experiment = next(
            item for item in self.batch.experiments if item.run_id == run_id
        )
        config = experiment._cfg
        paths = sorted(self._stage_paths(stage_index), key=repr)
        return (
            stage_index,
            config_observation(config, ("seed",)),
            tuple((path, config_observation(config, path)) for path in paths),
        )

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
            if details.get("probe") or not isinstance(stage_index, int):
                return None
            active[self._stage_group(run_id, stage_index)] = (run_id, stage_index)
        groups = dict(active)
        for run_id in queue:
            start = self.batch._stage_progress[run_id]
            experiment = next(
                item for item in self.batch.experiments if item.run_id == run_id
            )
            for stage_index in range(start, len(experiment.pipeline.stages)):
                group = self._stage_group(run_id, stage_index)
                if group not in completed and group not in groups:
                    groups[group] = (run_id, stage_index)
        return groups, set(active)

    def _stage_vectors(self, stage_index):
        """Encode only the rolling dependency prefix of a Stage's inputs."""
        paths = self._stage_paths(stage_index)
        configs = []
        for experiment in self.batch.experiments:
            cfg = experiment.cfg
            if not paths:
                # Before any dependency sample, use the whole configuration to
                # choose informative cold-start probes.
                configs.append(cfg)
                continue
            projected = {}
            for path in paths:
                value = cfg
                try:
                    for key in path:
                        value = value[key]
                except (KeyError, IndexError, TypeError):
                    value = None
                projected[repr(path)] = value
            configs.append(projected)
        return features(configs), {
            experiment.run_id: index
            for index, experiment in enumerate(self.batch.experiments)
        }

    def _stage_peak(self, run_id, stage_index, field, default):
        """Lazily fit one configuration-based peak model per top-level Stage."""
        key = (stage_index, field)
        model = self._stage_peak_models.get(key)
        if model is None:
            vectors, indices = self._stage_vectors(stage_index)
            model = LocalQuantileEstimator(
                vectors,
                indices,
                default,
                include_zero=field == "gpu_peak_kb",
            )
            for entry in self.batch._stage_history:
                if entry.get("stage") != stage_index or entry.get("reused"):
                    continue
                model.record(
                    entry.get("run_id"),
                    entry.get(field),
                    capped=entry.get(
                        "capped" if field == "peak_kb" else "gpu_capped", False
                    ),
                )
            self._stage_peak_models[key] = model
        return model.estimate(run_id)

    def _stage_duration(self, run_id, stage_index):
        """Estimate one Stage duration from its completed dependency samples.

        Models are cached and invalidated only when a Stage result arrives.  Tick
        processing merely reads the latest model; it never treats elapsed running
        time as a new sample.
        """
        model = self._stage_duration_models.get(stage_index)
        if model is None:
            vectors, indices = self._stage_vectors(stage_index)
            completed = []
            censored = []
            for entry in self.batch._stage_history:
                if entry.get("stage") != stage_index or entry.get("reused"):
                    continue
                entry_run_id = entry.get("run_id")
                row = indices.get(entry_run_id)
                duration = entry.get("stage_duration_seconds")
                if (
                    row is None
                    or not isinstance(duration, (int, float))
                    or duration <= 0
                ):
                    continue
                if entry.get("status") == Status.SUCCEEDED.value:
                    completed.append((entry_run_id, duration))
                elif entry.get("status") == Status.CANCELLED.value:
                    censored.append((entry_run_id, duration))
            if not completed:
                self._stage_duration_models[stage_index] = False
                return None
            model = LocalQuantileEstimator(
                vectors,
                indices,
                0.0,
                quantile=self.batch.estimate_coverage,
                # A cancelled duration is already a finite lower bound. Unlike
                # a cgroup peak it does not need a resource retry bump.
                margin=1.0,
            )
            for entry_run_id, duration in completed:
                model.record(entry_run_id, duration)
            for entry_run_id, duration in censored:
                model.record(entry_run_id, duration, capped=True)
            self._stage_duration_models[stage_index] = model
        if model is False:
            return None
        return model.estimate(run_id)

    @property
    def peak_ceiling(self):
        """Largest observed host peak, retained for scheduler introspection."""
        return self.peaks.ceiling_kb

    def _expected_peak(self, run_id):
        """Return the feature-based reservation for one pending run."""
        return self.peaks.estimate_kb(run_id)

    def _stage_candidates(self, stage_index, gpu_capacity_kb):
        """Return diverse ready candidates for one Stage and its own history only."""
        batch = self.batch
        run_ids = [
            run_id
            for run_id in batch._queue
            if batch._stage_progress[run_id] == stage_index
        ]
        if not self._probe_pending:
            # Dependency groups are known after the complete-pipeline probe.
            # Keep exactly one representative runnable; cache materialization
            # advances every other member once that representative finishes.
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
        vectors, indices = self._stage_vectors(stage_index)
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
            row = vectors[indices[run_id]][1:]
            return min(
                sum(
                    (left - right) ** 2
                    for left, right in zip(row, vectors[item][1:], strict=True)
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
                run_id, stage_index, "peak_kb", devices.HOST_PEAK_KB_DEFAULT
            )
            gpu_estimate = self._stage_peak(
                run_id, stage_index, "gpu_peak_kb", devices.GPU_PEAK_KB_DEFAULT
            )
            candidates.append(
                Candidate(
                    run_id,
                    stage_index,
                    host_estimate,
                    limits.cap_kb(host_estimate),
                    min(
                        gpu_estimate * devices.CAPACITY_FACTOR,
                        gpu_capacity_kb,
                    ),
                    self._stage_duration(run_id, stage_index),
                    1.0 - rank / denominator,
                )
            )
        return candidates

    @staticmethod
    def _interleave(groups):
        """Round-robin candidates so one Stage cannot consume beam search first."""
        items = []
        longest = max((len(group) for group in groups), default=0)
        for index in range(longest):
            for group in groups:
                if index < len(group):
                    items.append(group[index])
        return items

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
                    "expected_seconds": worker.expected_seconds,
                    "stage_index": worker.stage_index,
                    "probe": worker.probe,
                    "resident_kb": worker.resident_kb,
                    "peak_kb": worker.peak_kb,
                    "gpu_allocated_kb": worker.gpu_allocated_kb,
                    "gpu_reserved_kb": worker.gpu_reserved_kb,
                    "gpu_peak_allocated_kb": worker.gpu_peak_allocated_kb,
                    "gpu_peak_reserved_kb": worker.gpu_peak_reserved_kb,
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
                self.batch.recorder.record(message["event"])
            elif kind == "resource":
                self._resources(worker, message)
            elif kind == "finished" and isinstance(
                message.get("result"), (StageResult, AttemptResult)
            ):
                worker.finished = message["result"]
                dependencies = message.get("dependencies")
                worker.dependencies = (
                    dependencies if isinstance(dependencies, list) else None
                )
                worker.reused = message.get("reused") is True
                reports = message.get("stage_reports")
                worker.stage_reports = reports if isinstance(reports, list) else None
                resources = message.get("resources")
                if isinstance(resources, dict):
                    self._resources(worker, resources)

    @staticmethod
    def _resources(worker, message):
        current, peak = message.get("host_current_kb"), message.get("host_peak_kb")
        if all(
            isinstance(value, (int, float)) and value >= 0 for value in (current, peak)
        ):
            worker.resident_kb, worker.peak_kb = current, peak
        gpu = message.get("gpu")
        if isinstance(gpu, dict):
            for name in (
                "gpu_allocated_kb",
                "gpu_reserved_kb",
                "gpu_peak_allocated_kb",
                "gpu_peak_reserved_kb",
            ):
                value = gpu.get(name)
                if isinstance(value, (int, float)) and value >= 0:
                    setattr(worker, name, value)
        events = message.get("memory_events")
        if isinstance(events, dict):
            worker.memory_events = events
        directory = message.get("cgroup")
        if isinstance(directory, str) and worker.limit is not None:
            worker.limit = replace(worker.limit, cgroup=Path(directory))

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
        """Return the cgroup result last reported by the worker over IPC."""
        exit_events = {} if worker.limit is None else limits.events(worker.limit.cgroup)
        capped = (
            worker.cap_hit
            or limits.refused(worker.memory_events)
            or limits.refused(exit_events)
        )
        peak = worker.peak_kb
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
        for _run_id, worker in self.workers.items():
            reach = max(
                worker.peak_kb,
                0.0 if worker.limit is None else worker.limit.cap_kb,
            )
            needed += max(0.0, reach - worker.resident_kb)
        return devices.fits_reserve(host, needed)

    def _admits_gpu(self, memory, device, contract_kb):
        """Reserve only each worker's future PyTorch allocator growth."""
        needed = contract_kb
        for worker in self.workers.values():
            if worker.device != device:
                continue
            needed += max(0.0, worker.gpu_contract_kb - (worker.gpu_reserved_kb or 0))
        return memory.free * 1024 >= needed

    def _resource_left(self, memory, host):
        """Capacity available after reserving every running worker's future growth."""
        host_left = host.headroom_kb
        for worker in self.workers.values():
            reach = max(
                worker.peak_kb,
                0.0 if worker.limit is None else worker.limit.cap_kb,
            )
            host_left -= max(0.0, reach - worker.resident_kb)
        gpu_left = {}
        for device, observation in memory.items():
            gpu_left[device] = observation.free * 1024 - sum(
                max(0.0, worker.gpu_contract_kb - (worker.gpu_reserved_kb or 0))
                for worker in self.workers.values()
                if worker.device == device
            )
        return host_left, gpu_left

    @staticmethod
    def _plan_score(plan, host_capacity, gpu_capacities):
        """Prefer plans that jointly leave the least normalized resource behind."""
        unused = (plan.host_left_kb / host_capacity) ** 2
        unused += sum(
            (left / capacity) ** 2
            for left, capacity in zip(plan.gpu_left_kb, gpu_capacities, strict=True)
            if capacity > 0
        )
        # Parameter diversity only breaks materially equivalent packing choices.
        return -unused + 1e-6 * plan.novelty

    def _launch_plan(self, memory, host):
        """Choose a bounded multi-resource packing plan for this scheduler tick."""
        host_capacity, available_gpu = self._resource_left(memory, host)
        devices_in_plan = tuple(memory)
        if host_capacity <= 0 or not devices_in_plan:
            return ()
        gpu_capacities = tuple(
            max(0.0, available_gpu[device]) for device in devices_in_plan
        )
        stages = sorted(
            {self.batch._stage_progress[run_id] for run_id in self.batch._queue}
        )
        gpu_capacity_kb = min(memory[device].total * 1024 for device in devices_in_plan)
        candidates = self._interleave(
            [self._stage_candidates(stage, gpu_capacity_kb) for stage in stages]
        )
        plans = [Plan(host_capacity, gpu_capacities)]
        for candidate in candidates:
            expanded = list(plans)
            for plan in plans:
                if candidate.host_contract_kb > plan.host_left_kb:
                    continue
                for index, gpu_left in enumerate(plan.gpu_left_kb):
                    if candidate.gpu_contract_kb > gpu_left:
                        continue
                    left = list(plan.gpu_left_kb)
                    left[index] -= candidate.gpu_contract_kb
                    expanded.append(
                        Plan(
                            plan.host_left_kb - candidate.host_contract_kb,
                            tuple(left),
                            (*plan.placements, (candidate, devices_in_plan[index])),
                            plan.novelty + candidate.novelty,
                        )
                    )
            plans = sorted(
                expanded,
                key=lambda plan: self._plan_score(plan, host_capacity, gpu_capacities),
                reverse=True,
            )[:PLAN_BEAM_WIDTH]
        return max(
            plans,
            key=lambda plan: self._plan_score(plan, host_capacity, gpu_capacities),
        ).placements

    def _launch_available(self, memory, host):
        """Launch the resource-aware packing plan chosen from this tick's snapshot."""
        for candidate, device in self._launch_plan(memory, host):
            self._launch(device, memory[device], host, candidate)

    def _probe_candidate(self, memory, host):
        """Give the dependency probe this tick's entire allocatable capacity."""
        device, observation = max(memory.items(), key=lambda item: item[1].free)
        host_estimate = host.headroom_kb / devices.CAPACITY_FACTOR
        host_contract = limits.cap_kb(host_estimate)
        # ``cap_kb`` has a startup floor.  Do not let that floor turn a tiny
        # headroom into an overcommitted probe.
        if host_contract > host.headroom_kb:
            return None
        return device, Candidate(
            self.batch._queue[0],
            0,
            host_estimate,
            host_contract,
            observation.free * 1024,
            None,
            1.0,
        )

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
                candidate = shared.find(parent, (stage_index,), experiment._cfg)
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
                self.batch._stage_history.append(
                    {
                        "run_id": run_id,
                        "stage": stage_index,
                        "attempt": attempt,
                        "status": Status.SUCCEEDED.value,
                        "reused": True,
                        "dependencies": completed.get("config_dependencies"),
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
                break
            if not advanced:
                break
        if changed:
            self.batch._save()
        return changed

    def _launch(self, device, memory, host, candidate, *, probe=False):
        batch = self.batch
        with batch._state_lock:
            if candidate.run_id not in batch._queue:
                return False
            batch._queue.remove(candidate.run_id)
        run_id = candidate.run_id
        stage_index = None if probe else candidate.stage_index
        host_estimate = candidate.host_estimate_kb
        host_contract = candidate.host_contract_kb
        gpu_contract = candidate.gpu_contract_kb
        duration_estimate = candidate.expected_seconds
        gpu_fits = self._admits_gpu(memory, device, gpu_contract)
        if not self._admits(host, host_contract) or not gpu_fits:
            # Put the selection back at the front. Dispatching updated the
            # estimator's own state, and re-selecting the same run is idempotent,
            # so the queue keeps its remaining order while the host is short.
            with batch._state_lock:
                batch._queue.appendleft(run_id)
            return False
        with batch._state_lock:
            batch._active_gpu[run_id] = device
        experiment = next(item for item in batch.experiments if item.run_id == run_id)
        # A full-Pipeline probe is also the first execution attempt of Stage 0.
        # If it is interrupted, Stage 0 must resume as attempt 2 so user
        # checkpoint/retry logic does not mistake the recovery for a new run.
        stage_key = f"{run_id}:{0 if probe else stage_index}"
        previous_stage_attempt = batch._stage_attempts.get(stage_key)
        stage_attempt = (previous_stage_attempt or 0) + 1
        root = (
            experiment.output_dir
            / "attempts"
            / ("probe" if probe else f"stage-{stage_index}")
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
                        replace(item, _output=None, output_source=None)
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
            # The unit is unique per attempt: systemd unloads a finished scope
            # asynchronously, and a retry must not collide with the previous one.
            limit = limits.memory_limit(
                f"expman-{run_id[:12]}-{'probe' if probe else stage_index}-{stage_attempt}",
                host_estimate,
            )
            if limit.cgroup is not None:
                env["EXPMAN_CGROUP"] = str(limit.cgroup)
            parent_socket, child_socket = socket.socketpair()
            parent_socket.setblocking(False)
            env["EXPMAN_IPC_FD"] = str(child_socket.fileno())
            env["EXPMAN_POLL_INTERVAL"] = str(devices.POLL_INTERVAL)
            if gpu_contract > 0:
                env["EXPMAN_GPU_LIMIT_KB"] = str(gpu_contract)
            self._account()
            with (root / "output.log").open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    [*limit.command, sys.executable, "-m", "expman._worker", str(root)],
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
            if limit.mechanism == "systemd":
                # ``systemd-run --scope`` has placed this process in a user
                # hierarchy by now.  Discover it from proc rather than deriving
                # a path from the scheduler's unrelated cgroup.
                limit = replace(limit, cgroup=limits.cgroup_for_pid(process.pid))
            now = perf_counter()
            with experiment._timing_lock:
                experiment._active_started = now
            self.workers[run_id] = Worker(
                process=process,
                experiment=experiment,
                device=device,
                uuid=None if memory is None else memory.uuid,
                root=root,
                started=now,
                stage_index=stage_index,
                stage_attempt=stage_attempt,
                channel=Channel(parent_socket),
                gpu_contract_kb=gpu_contract,
                expected_seconds=duration_estimate,
                limit=limit,
                probe=probe,
            )
            parent_socket = None
        except BaseException:
            if child_socket is not None:
                child_socket.close()
            if parent_socket is not None:
                parent_socket.close()
            if process is not None:
                _kill_group(process)
                process.wait()
                process.stdin.close()
                self.workers.pop(run_id, None)
                with experiment._timing_lock:
                    experiment._active_started = None
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
        _kill_group(worker.process)  # Also reap experiment-owned helper processes.
        worker.process.wait()
        worker.process.stdin.close()
        self._messages(worker)
        experiment = worker.experiment
        duration = max(0.0, finished - worker.started)
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
        cuda_out_of_memory = devices.cuda_out_of_memory(
            result.error_type, result.error_message
        )
        gpu_capped = cuda_out_of_memory
        if gpu_capped and cancelled is None and result.status is not Status.SUCCEEDED:
            # CUDA uses the same error for a process allocator limit and an
            # external allocation refusal. Both mean this Stage needs a larger
            # contract on its next attempt; resource admission will decide when
            # that expanded contract fits again.
            result = replace(
                result,
                status=Status.CANCELLED,
                error_type="GpuMemoryLimit",
                error_message=(
                    "attempt reached its PyTorch memory limit of "
                    f"{worker.gpu_contract_kb / 1024:.0f} MiB"
                ),
            )
        if worker.probe:
            with self.batch._state_lock:
                experiment._active_started = None
                # The probe executes Stage 0 as part of the complete Pipeline.
                # Preserve that attempt number even when the probe is interrupted:
                # the first normal Stage-0 worker must resume as attempt 2.
                self.batch._stage_attempts[f"{run_id}:0"] = worker.stage_attempt
                self.batch._active_gpu.pop(run_id)
                self.batch._gpu_running_info.pop(run_id, None)
                del self.workers[run_id]
                if worker.channel is not None:
                    worker.channel.socket.close()
                self._probe_pending = False
                if result.status is Status.SUCCEEDED:
                    for report in worker.stage_reports or ():
                        if not isinstance(report, dict) or not isinstance(
                            report.get("stage"), int
                        ):
                            continue
                        self.batch._stage_history.append(
                            {
                                "run_id": run_id,
                                "stage": report["stage"],
                                "status": Status.SUCCEEDED.value,
                                "reused": True,
                                "probe": True,
                                "dependencies": report.get("dependencies"),
                            }
                        )
                    experiment._attempts.append(
                        AttemptResult(
                            run_id,
                            len(experiment.result.attempts) + 1,
                            Status.SUCCEEDED,
                            duration,
                            output_source=SnapshotOutput(
                                experiment._store,
                                (len(experiment.pipeline.stages) - 1,),
                            )
                            if experiment.pipeline.stages
                            else None,
                        )
                    )
                else:
                    experiment._attempts.append(
                        AttemptResult(
                            run_id,
                            len(experiment.result.attempts) + 1,
                            result.status,
                            duration,
                            error_type=result.error_type,
                            error_message=result.error_message,
                        )
                    )
                    self.batch._queue.appendleft(run_id)
                self.batch._save()
            return
        with self.batch._state_lock:
            experiment._active_started = None
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
                # An OOM proves the demand exceeded the attempted contract even
                # when PyTorch could not report its final allocator counters.
                "gpu_peak_kb": max(
                    worker.gpu_peak_reserved_kb or 0.0,
                    (
                        max(worker.gpu_contract_kb, devices.GPU_PEAK_KB_DEFAULT)
                        if gpu_capped
                        else 0.0
                    ),
                ),
                "gpu_capped": gpu_capped,
                "capped": capped,
                "reused": worker.reused,
                "dependencies": worker.dependencies,
            }
            self.batch._stage_history.append(observation)
            self._stage_peak_models.pop((worker.stage_index, "peak_kb"), None)
            self._stage_peak_models.pop((worker.stage_index, "gpu_peak_kb"), None)
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
            if cancelled != "MemoryPressure" and not capped:
                # A natural exit frees host pressure for subsequent admission.
                self.host_gate.mem_block = False
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
                launching = self.host_gate.can_launch(host, list(self.workers))
                if launching and memory is not None:
                    if self._probe_pending and not self.workers and self.batch._queue:
                        probe = self._probe_candidate(memory, host)
                        if probe is not None:
                            device, candidate = probe
                            self._launch(
                                device, memory[device], host, candidate, probe=True
                            )
                    elif not self._probe_pending:
                        self._materialize_reuses()
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
