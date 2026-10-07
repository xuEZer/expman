"""批次收尾：清掉只服务于「中断后恢复」、且能由重跑重建的中间态。

为什么值得做：`cache/` 存的是各 Stage 的输出快照，本机 D001 实测 **105 MB**，
而真正是结果的东西（`metrics.sqlite3`、`output.log`、`config.pkl`）只有 4 MB 量级。

只在**批次正常跑完**之后清理。中断或异常退出会保留现场，`resume` 照常可用——
这正是 resume 存在的意义，不该被收尾逻辑误伤。

清理后批次不能再 resume：`Batch.resume` 会先看到 `CACHE_CLEANED` 并给出明确报错，
而不是等到 `successful experiment has no final snapshot` 这种绕远的失败。
"""

from __future__ import annotations

import shutil
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

CACHE_DIR = "cache"
EXPERIMENTS_DIR = "experiments"
SENTINEL = "CACHE_CLEANED"

# 只服务于恢复，能由重跑重建 ⇒ 删
RUN_FILES = ("rng_initial.pkl",)
STAGE_FILES = ("completed.pkl", "pipeline.pkl")
ATTEMPT_FILES = ("bootstrap.pkl", "input.pkl")

# 明确保留（这些是批次的记录，不是缓存）：
#   batch.pkl / metrics.sqlite3 / timing.pkl / live.json / run.lock
#   experiments/<run>/config.pkl
#   experiments/<run>/attempts/<stage>/<n>/output.log
#   experiments/<run>/stages/<i>/status.pkl


@dataclass
class CleanReport:
    """清理结果。`bytes` 含 shared cache；`cache_bytes` 单独列出便于对账。"""

    files: int = 0
    bytes: int = 0
    cache_bytes: int = 0

    @property
    def empty(self) -> bool:
        return self.files == 0

    def summary(self) -> str:
        return (
            f"{self.files} 个文件 / {self.bytes / 1e6:.1f} MB"
            f"（其中 shared cache {self.cache_bytes / 1e6:.1f} MB）"
        )


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _tree_files(root: Path) -> list[Path]:
    try:
        return [path for path in root.rglob("*") if path.is_file()]
    except OSError:
        return []


def clean(output_dir: str | Path) -> CleanReport:
    """删除可重算的中间态，返回删掉的文件数与字节数。

    幂等：目录已经清过时返回空报告，不会写 sentinel。
    """
    root = Path(output_dir)
    report = CleanReport()

    cache = root / CACHE_DIR
    if cache.is_dir():
        files = _tree_files(cache)
        report.cache_bytes = sum(_size(path) for path in files)
        report.files += len(files)
        report.bytes += report.cache_bytes
        shutil.rmtree(cache, ignore_errors=True)

    experiments = root / EXPERIMENTS_DIR
    if experiments.is_dir():
        stray: list[Path] = []
        for run_dir in experiments.iterdir():
            if not run_dir.is_dir():
                continue
            stray += [run_dir / name for name in RUN_FILES]
            for parent_name, names in (("stages", STAGE_FILES), ("attempts", None)):
                parent = run_dir / parent_name
                if not parent.is_dir():
                    continue
                if names is not None:
                    stray += [
                        child / name
                        for child in parent.iterdir()
                        if child.is_dir()
                        for name in names
                    ]
                else:  # attempts/<stage>/<n>/
                    stray += [
                        number_dir / name
                        for stage_dir in parent.iterdir()
                        if stage_dir.is_dir()
                        for number_dir in stage_dir.iterdir()
                        if number_dir.is_dir()
                        for name in ATTEMPT_FILES
                    ]
        for path in stray:
            if path.is_file():
                report.bytes += _size(path)
                report.files += 1
                with suppress(OSError):
                    path.unlink()

    if report.files:
        (root / SENTINEL).write_text(
            "这个批次正常跑完后清掉了可重算的中间态（shared cache 与 attempt/stage 快照），\n"
            "所以它不能再 resume。要保留现场，重跑时加 --keep-cache。\n\n"
            f"清理结果：{report.summary()}\n",
            encoding="utf-8",
        )
    return report


def was_cleaned(output_dir: str | Path) -> bool:
    """批次是否已经收尾清理过（⇒ 不可 resume）。"""
    return (Path(output_dir) / SENTINEL).is_file()
