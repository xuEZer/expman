"""Reclaim a physical prefix once every possible consumer has passed it."""

from collections import defaultdict
from dataclasses import dataclass

from expman.core.events import Status


@dataclass
class _Node:
    owner: str
    stage: int
    members: list[str]
    next_member: int = 0


class PrefixReclaimer:
    def __init__(self, stages):
        self.stages = stages
        self.batch = stages.batch
        self._members_by_stage = {}
        self._paths = defaultdict(list)
        self._private_stage = {}
        self._nodes = set()
        self._waiting = defaultdict(list)

    def _members(self, run_id, stage):
        if stage not in self._members_by_stage:
            groups = defaultdict(list)
            for member in self.batch.by_id:
                groups[self.stages.group(member, stage)].append(member)
            self._members_by_stage[stage] = groups
        return self._members_by_stage[stage][self.stages.group(run_id, stage)]

    def attempt_started(self, run_id):
        # A retry can publish its own new physical node before the scheduler
        # commits its success. Other runs may attach to this node immediately.
        self._private_stage.pop(run_id, None)

    def reclaim_after(self, run_id, stage, *, succeeded=True):
        """Revisit readers after a durable success or a private, unfinished attempt."""
        store = self.batch.by_id[run_id]._store
        position = (stage,)
        if succeeded:
            reference = store.completed_reference(position)
            self._paths[run_id].append(reference)
            self._private_stage.pop(run_id, None)
        else:
            # A completed local record can still reference the same physical
            # node even if the worker exited before acknowledging its success.
            if store.can_reuse(position) or store.has_completed(position):
                return
            self._private_stage[run_id] = stage
            reference = None
        checks = self._waiting.pop(run_id, [])
        if reference is not None and reference not in self._nodes:
            self._nodes.add(reference)
            checks.append(_Node(run_id, stage, self._members(run_id, stage)))
        for node in checks:
            self._reclaim_or_wait(node)

    def _reclaim_or_wait(self, node):
        owner_path = self._paths[node.owner]
        leaf = node.stage == len(self.batch.pipeline.stages) - 1
        # A reader needs a non-leaf until its child succeeds. A leaf's
        # metadata can be used until the reader commits its own leaf.
        needed = node.stage + (1 if leaf else 2)
        while node.next_member < len(node.members):
            member = node.members[node.next_member]
            path = self._paths[member]
            depth = min(len(path), node.stage + 1)
            private_stage = self._private_stage.get(member)
            # UUIDs identify physical nodes. One different ancestor rules out
            # this entire branch. So does a stage this run must compute itself.
            if (
                len(path) < needed
                and (not depth or path[depth - 1] == owner_path[depth - 1])
                and (private_stage is None or private_stage > node.stage)
            ):
                self._waiting[member].append(node)
                return
            node.next_member += 1

        reference = owner_path[node.stage]
        self.batch.by_id[node.owner]._store.shared.reclaim(reference, leaf=leaf)
        self._nodes.remove(reference)

    def reclaim_saved(self):
        """Rebuild readers and retry cleanup from the committed run cursors."""
        for run_id, cursor in self.batch.state.stage_progress.items():
            succeeded = self.batch.by_id[run_id].result.status is Status.SUCCEEDED
            for stage in range(cursor + int(succeeded)):
                self.reclaim_after(run_id, stage)
            if not succeeded:
                self.reclaim_after(run_id, cursor, succeeded=False)
