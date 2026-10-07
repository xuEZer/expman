"""批次收尾：删掉 `<batch>/cache/`。

`cache/` 是跨 Run 复用的 Stage 输出前缀缓存（`snapshot.pkl` / `metadata.pkl`），
只服务于「同一批次内多个 Run 共享上游 Stage」，跑完即无价值。实测一次 13 个 Run
的批次里它占 **105 MB**，而批次目录里其余全部东西（`batch.pkl` / `metrics.sqlite3` /
`config.pkl` / `output.log`）加起来只有 **4 MB** 量级——`runs/` 的体积几乎全在这里。

**只删 cache，不删 attempt / stage 中间态。** 后者体积上几乎没有收益，却是 load-bearing
的：`Batch.resume` 需要它（`completed.pkl` 是「成功实验的最终快照」，缺了直接报错），
框架也把「结果从磁盘回读而不是常驻内存」当作一条设计属性。

在哪里被调用
------------
`cli._execute()`，批次正常跑完、结果已经打印出来之后。

为什么**不**放在 `Batch.run()` 里：`AttemptResult.output` 是惰性的——每次访问都从权威
记录重读（见 `experiment.py` 的类注释），而共享前缀成员的记录就在 `<batch>/cache/`
下面。`run()` 一返回就删，会让本进程再也读不出结果，`Batch.results` 直接抛
`StorageError`。放到 CLI 层后：

- 进程内语义不变：`batch.run()` 返回后结果照样读得出来（框架自己的测试就是靠这个）；
- 真正的跑批入口 `expman run` 在打印完结果后丢掉缓存，`runs/` 的体积收益照样拿到；
- 中断 / 异常路径不会走到这里，现场原样留给 `resume` 与排查。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

CACHE_DIR = "cache"


@dataclass
class CleanReport:
    """清理结果：删掉的文件数与字节数。"""

    files: int = 0
    bytes: int = 0

    @property
    def empty(self) -> bool:
        return self.files == 0

    def summary(self) -> str:
        return f"{self.files} 个文件 / {self.bytes / 1e6:.1f} MB"


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def clean(output_dir: str | Path) -> CleanReport:
    """删除 `<output_dir>/cache/`，返回删掉的文件数与字节数。

    幂等：缓存不存在时返回空报告。
    """
    cache = Path(output_dir) / CACHE_DIR
    if not cache.is_dir():
        return CleanReport()
    try:
        files = [path for path in cache.rglob("*") if path.is_file()]
    except OSError:
        files = []
    report = CleanReport(files=len(files), bytes=sum(_size(path) for path in files))
    shutil.rmtree(cache, ignore_errors=True)
    return report
