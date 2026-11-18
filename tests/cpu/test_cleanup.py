from pathlib import Path

import pytest

from expman import cleanup
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
