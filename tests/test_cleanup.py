"""批次收尾：删哪些中间态、留哪些记录、以及不可恢复的判定。"""

from pathlib import Path

from expman import cleanup

CACHE_KEY = "a" * 64
RUN_ID = "b" * 32

# 收尾必须删掉的（只服务于 resume）
DROPPED = (
    "experiments/{run}/rng_initial.pkl",
    "experiments/{run}/stages/0/completed.pkl",
    "experiments/{run}/stages/1/pipeline.pkl",
    "experiments/{run}/attempts/stage-0/1/bootstrap.pkl",
    "experiments/{run}/attempts/stage-0/1/input.pkl",
    "experiments/{run}/attempts/stage-1/1/bootstrap.pkl",
    "experiments/{run}/attempts/stage-1/1/input.pkl",
)

# 收尾必须保留的（批次记录，不是缓存）
KEPT = (
    "batch.pkl",
    "metrics.sqlite3",
    "timing.pkl",
    "live.json",
    "experiments/{run}/config.pkl",
    "experiments/{run}/stages/0/status.pkl",
    "experiments/{run}/stages/1/status.pkl",
    "experiments/{run}/attempts/stage-0/1/output.log",
    "experiments/{run}/attempts/stage-1/1/output.log",
)

CACHE_FILE_BYTES = 40  # 32 + 8


def make_batch(root: Path) -> Path:
    """搭一棵含全部相关文件的最小批次目录树。"""
    cache_leaf = root / "cache" / CACHE_KEY / "root" / RUN_ID
    cache_leaf.mkdir(parents=True)
    (cache_leaf / "snapshot.pkl").write_bytes(b"x" * 32)
    (cache_leaf / "metadata.pkl").write_bytes(b"x" * 8)

    payloads = {
        "batch.pkl": b"manifest",
        "metrics.sqlite3": b"metrics",
        "timing.pkl": b"timing",
        "live.json": b"{}",
        "experiments/{run}/config.pkl": b"cfg",
        "experiments/{run}/stages/0/status.pkl": b"status",
        "experiments/{run}/stages/1/status.pkl": b"status",
        "experiments/{run}/attempts/stage-0/1/output.log": b"log",
        "experiments/{run}/attempts/stage-1/1/output.log": b"log",
        "experiments/{run}/rng_initial.pkl": b"rng",
        "experiments/{run}/stages/0/completed.pkl": b"snap",
        "experiments/{run}/stages/1/pipeline.pkl": b"pipe",
        "experiments/{run}/attempts/stage-0/1/bootstrap.pkl": b"boot",
        "experiments/{run}/attempts/stage-0/1/input.pkl": b"in",
        "experiments/{run}/attempts/stage-1/1/bootstrap.pkl": b"boot",
        "experiments/{run}/attempts/stage-1/1/input.pkl": b"in",
    }
    for rel, payload in payloads.items():
        path = root / rel.format(run=RUN_ID)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return root


def test_clean_drops_resume_state_and_keeps_records(tmp_path):
    root = make_batch(tmp_path)

    report = cleanup.clean(root)

    assert report.files == len(DROPPED) + 2  # 中间态 + 两个 cache 文件
    assert report.cache_bytes == CACHE_FILE_BYTES
    assert report.bytes == CACHE_FILE_BYTES + 23  # 32+8 cache 与各中间态的字节和
    assert not (root / "cache").exists()
    for rel in DROPPED:
        assert not (root / rel.format(run=RUN_ID)).exists(), rel
    for rel in KEPT:
        assert (root / rel.format(run=RUN_ID)).is_file(), rel
    assert cleanup.was_cleaned(root)


def test_clean_is_idempotent(tmp_path):
    root = make_batch(tmp_path)

    assert cleanup.clean(root).files > 0
    sentinel = root / cleanup.SENTINEL
    first = sentinel.read_text(encoding="utf-8")

    second = cleanup.clean(root)

    assert second.empty
    assert second.bytes == 0
    assert sentinel.read_text(encoding="utf-8") == first


def test_clean_of_a_pristine_batch_leaves_no_sentinel(tmp_path):
    root = make_batch(tmp_path)
    (root / "cache").rename(tmp_path / "moved-cache")
    for rel in DROPPED:
        (root / rel.format(run=RUN_ID)).unlink()

    report = cleanup.clean(root)

    assert report.empty
    assert not cleanup.was_cleaned(root)


def test_was_cleaned_is_false_before_finalization(tmp_path):
    root = make_batch(tmp_path)

    assert not cleanup.was_cleaned(root)
