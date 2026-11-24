from pathlib import Path

import pytest
from helpers.batches import make_batch, restore_batch

from expman import Pipeline, Stage, Status, cleanup
from expman.scheduling._prefixes import PrefixReclaimer
from expman.scheduling._stages import StageWork
from expman.storage.base import PickleSerializer, StorageError, run_lock, write_record

KEY = "a" * 64
USED = (KEY, "root", "b" * 32)
ORPHAN = (KEY, "root", "c" * 32)


def make_records(root: Path) -> Path:
    stage = root / "experiments" / "run-1" / "stages" / "0" / "completed.pkl"
    write_record(stage, {"cache_ref": USED}, PickleSerializer())
    for node in (USED, ORPHAN):
        directory = root.joinpath("cache", *node)
        directory.mkdir(parents=True)
        (directory / "snapshot.pkl").write_bytes(b"abc")
        (directory / "metadata.pkl").write_bytes(b"x")
    return stage


def test_cleanup_keeps_nodes_used_by_completed_stages(tmp_path):
    stage = make_records(tmp_path)
    report = cleanup.clean(tmp_path)
    assert (report.files, report.bytes) == (2, 4)
    assert not tmp_path.joinpath("cache", *ORPHAN).exists()
    assert tmp_path.joinpath("cache", *USED, "snapshot.pkl").read_bytes() == b"abc"
    assert stage.is_file()
    assert cleanup.clean(tmp_path).empty


def test_unreadable_references_abort_before_deletion(tmp_path):
    make_records(tmp_path).write_bytes(b"unreadable")
    with pytest.raises(StorageError):
        cleanup.clean(tmp_path)
    assert tmp_path.joinpath("cache", *ORPHAN, "snapshot.pkl").is_file()


def test_cleanup_refuses_an_active_batch(tmp_path):
    make_records(tmp_path)
    with run_lock(tmp_path), pytest.raises(StorageError, match="already running"):
        cleanup.clean(tmp_path)
    assert tmp_path.joinpath("cache", *ORPHAN).is_dir()


def test_missing_cache_requires_no_cleanup(tmp_path):
    assert cleanup.clean(tmp_path).empty


@pytest.mark.parametrize("reopen", [False, True])
def test_failed_child_retains_only_its_own_physical_branch(tmp_path, reopen):
    class Source(Stage):
        @classmethod
        def config_dependencies(cls, cfg):
            return False

        def process(self, data, ctx):
            return 1

    class Child(Source):
        failed_run = None

        def process(self, data, ctx):
            if ctx.run_id == self.failed_run:
                raise ValueError("failed child")

    config = tmp_path / "config.yaml"
    config.write_text("device: [0]\nseed: 0\nitem: 0\ngrid:\n  item: [1, 2]\n")
    pipeline = Pipeline([Source, Child, Source, Child])
    batch = make_batch(pipeline, config, output_dir=tmp_path / "batch")
    first, second = batch.experiments
    Child.failed_run = second.run_id
    reclaimer = PrefixReclaimer(StageWork(batch))
    cache = batch.output_dir / "cache"
    first_nodes = []
    for stage in range(len(pipeline.stages)):
        assert first.run_stage(stage, attempt=1).status is Status.SUCCEEDED
        reclaimer.reclaim_after(first.run_id, stage)
        first_nodes.append(cache.joinpath(*first._store.completed_reference((stage,))))
    # Even the deepest node could still supply the second run.
    assert all(node.is_dir() for node in first_nodes)

    # A prior interrupted attempt forces this run onto a different root node.
    second._store.status((0,), Status.CANCELLED.value, 1)
    assert second.run_stage(0, attempt=2).status is Status.SUCCEEDED
    second_node = cache.joinpath(*second._store.completed_reference((0,)))
    assert second_node != first_nodes[0]
    assert second.run_stage(1, attempt=1).status is Status.FAILED
    if reopen:
        # Restart between committing the assignments and reclaiming the nodes.
        batch.state.stage_progress[first.run_id] = len(pipeline.stages) - 1
        batch.state.stage_progress[second.run_id] = 1
        first.record_attempt(Status.SUCCEEDED, 1.0)
        second.record_attempt(Status.FAILED, 1.0)
        batch.state.queue.clear()
        batch._save()
        restore_batch(pipeline, batch.output_dir).run()
    else:
        reclaimer.reclaim_after(second.run_id, 0)

    assert all(not node.exists() for node in first_nodes)
    assert (second_node / "snapshot.pkl").is_file()


class SharedStep(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return False

    def process(self, data, ctx):
        return None


def matching_runs(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("device: [0]\nseed: 0\nitem: 0\ngrid:\n  item: [1, 2]\n")
    pipeline = Pipeline([SharedStep, SharedStep, SharedStep])
    return make_batch(pipeline, config, output_dir=tmp_path / "batch")


@pytest.mark.parametrize("reopen", [False, True])
@pytest.mark.parametrize("completed_on_disk", [False, True])
def test_failed_reader_keeps_its_input_and_any_locally_attached_node(
    tmp_path, reopen, completed_on_disk
):
    batch = matching_runs(tmp_path)
    first, second = batch.experiments
    reclaimer = PrefixReclaimer(StageWork(batch))
    nodes = []
    for stage in range(3):
        assert first.run_stage(stage, attempt=1).status is Status.SUCCEEDED
        reclaimer.reclaim_after(first.run_id, stage)
        nodes.append(
            (batch.output_dir / "cache").joinpath(
                *first._store.completed_reference((stage,))
            )
        )
    assert second.run_stage(0, attempt=1).status is Status.SUCCEEDED
    reclaimer.reclaim_after(second.run_id, 0)
    assert all(node.is_dir() for node in nodes)
    if completed_on_disk:
        # The child attached the shared node before its final IPC message was lost.
        assert second.run_stage(1, attempt=1).status is Status.SUCCEEDED
    else:
        # This attempt entered its own computation and exited before publication.
        second._store.status((1,), Status.FAILED.value, 1)
    if reopen:
        batch.state.stage_progress[first.run_id] = 2
        batch.state.stage_progress[second.run_id] = 1
        first.record_attempt(Status.SUCCEEDED, 1.0)
        second.record_attempt(Status.FAILED, 1.0)
        batch.state.queue.clear()
        batch._save()
        restore_batch(batch.pipeline, batch.output_dir).run()
    else:
        reclaimer.reclaim_after(second.run_id, 1, succeeded=False)

    assert nodes[0].is_dir()  # A failed child can retry from its parent's output.
    assert [node.is_dir() for node in nodes[1:]] == [completed_on_disk] * 2


def test_active_private_retry_can_publish_a_node_before_its_success_is_committed(
    tmp_path,
):
    batch = matching_runs(tmp_path)
    first, second = batch.experiments
    reclaimer = PrefixReclaimer(StageWork(batch))
    second._store.status((0,), Status.FAILED.value, 1)
    reclaimer.reclaim_after(second.run_id, 0, succeeded=False)

    reclaimer.attempt_started(second.run_id)
    assert second.run_stage(0, attempt=2).status is Status.SUCCEEDED
    reference = second._store.completed_reference((0,))
    # The peer reuses the retry's node before the scheduler observes its exit.
    for stage in (0, 1):
        assert first.run_stage(stage, attempt=1).status is Status.SUCCEEDED
        reclaimer.reclaim_after(first.run_id, stage)
    assert first._store.completed_reference((0,)) == reference
    assert (batch.output_dir / "cache").joinpath(*reference, "snapshot.pkl").is_file()
