"""批次收尾：只清共享前缀缓存，不碰 resume 与结果回读所依赖的东西。"""

from pathlib import Path

from expman import cleanup

CACHE_KEY = "a" * 64
RUN_ID = "b" * 32
CACHE_BYTES = 40  # snapshot.pkl 32 + metadata.pkl 8

# 收尾必须保留的：resume（completed.pkl）与「结果从磁盘回读」（bootstrap/input）都依赖它们
KEPT = (
    "batch.pkl",
    "metrics.sqlite3",
    "timing.pkl",
    "live.json",
    "experiments/{run}/config.pkl",
    "experiments/{run}/rng_initial.pkl",
    "experiments/{run}/stages/0/status.pkl",
    "experiments/{run}/stages/0/completed.pkl",
    "experiments/{run}/stages/1/pipeline.pkl",
    "experiments/{run}/attempts/stage-0/1/bootstrap.pkl",
    "experiments/{run}/attempts/stage-0/1/input.pkl",
    "experiments/{run}/attempts/stage-0/1/output.log",
    "experiments/{run}/attempts/stage-1/1/output.log",
)


def make_batch(root: Path) -> Path:
    """搭一棵含 cache 与各类批次记录的目录树。"""
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
        "experiments/{run}/rng_initial.pkl": b"rng",
        "experiments/{run}/stages/0/status.pkl": b"status",
        "experiments/{run}/stages/0/completed.pkl": b"snap",
        "experiments/{run}/stages/1/pipeline.pkl": b"pipe",
        "experiments/{run}/attempts/stage-0/1/bootstrap.pkl": b"boot",
        "experiments/{run}/attempts/stage-0/1/input.pkl": b"in",
        "experiments/{run}/attempts/stage-0/1/output.log": b"log",
        "experiments/{run}/attempts/stage-1/1/output.log": b"log",
    }
    for rel, payload in payloads.items():
        path = root / rel.format(run=RUN_ID)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return root


def test_clean_drops_only_the_shared_cache(tmp_path):
    root = make_batch(tmp_path)

    report = cleanup.clean(root)

    assert report.files == 2
    assert report.bytes == CACHE_BYTES
    assert not (root / "cache").exists()
    for rel in KEPT:
        assert (root / rel.format(run=RUN_ID)).is_file(), rel


def test_clean_is_idempotent(tmp_path):
    root = make_batch(tmp_path)

    assert cleanup.clean(root).files == 2

    second = cleanup.clean(root)

    assert second.empty
    assert second.bytes == 0


def test_clean_of_a_batch_without_cache_is_a_noop(tmp_path):
    (tmp_path / "batch.pkl").write_bytes(b"manifest")

    report = cleanup.clean(tmp_path)

    assert report.empty
    assert (tmp_path / "batch.pkl").is_file()
