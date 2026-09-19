# 显式分层配置依赖

状态：已实现 · 日期：2026-09-19

关联：[DESIGN.md](../DESIGN.md) §3/§5/§7、[README.md](../README.md)「同 Batch 连续前缀复用」、
`src/expman/stage.py`、`src/expman/dependencies.py`、`src/expman/scheduling.py`。

## 1. 目标

让每个 Stage 显式声明它依赖哪些配置位置，并支持按配置取值选择不同分支；调度分组、
前缀复用、checkpoint 校验和估计特征全部以声明为唯一来源。

同时移除两套旧机制：

- **运行时读取追踪**：`ConfigurationReads`、`FrozenDict`/`FrozenList` 的 tracker 与
  checkpoint 依赖信封不再采集实际读取。
- **完整 Pipeline 探查（probe）**：调度器不再先跑一次整条 Pipeline 来发现依赖，
  依赖在派发前即可由声明计算。

## 2. 声明格式

`Stage.config_dependencies(cfg)` 是类方法，返回与配置树同构的声明：

- `True`：依赖该节点整棵子树；
- 映射：递归到键（映射用字符串键，序列用整数下标）；
- `False` 或缺省：不依赖该节点；
- 根部返回 `True`：依赖完整配置。

```python
class Train(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        paths = {"data": True, "model": {"name": True, "lr": True}}
        if cfg["Baseline"] == "B2":
            paths["predictor"] = True
        return paths
```

- `cfg` 是当前尝试的只读视图，因此分支可以按取值决定依赖。
- 默认实现返回 `True`：读取 `ctx.cfg` 却未重写该方法的 Stage 保守依赖完整配置，
  不会错误复用；不读取 `ctx.cfg` 的 Stage 可显式返回 `{}`。
- 声明解析由 `dependencies.declared_paths()` 完成，非法节点/键类型抛出 `TypeError`。

## 3. 解析与使用

`dependencies.py` 提供：

| 函数 | 作用 |
|---|---|
| `declared_paths(declaration)` | 把分层声明解析为去重排序后的路径元组 |
| `stage_dependencies(stage, config)` | 解析声明并计算每条路径的值摘要 |
| `observation(config, path)` | 路径取值的规范化 sha256 摘要 |
| `matches(dependencies, config)` | 快照依赖是否与当前配置一致 |
| `validate(dependencies)` | 持久化记录的格式校验 |

摘要计算先把 `FrozenDict`/`FrozenList`/集合规范化为普通映射、序列和 `frozenset`，
因此同一个配置无论以普通字典还是只读视图参与计算，摘要都一致。

使用位置：

- **Pipeline**（`pipeline.py`）：阶段执行前调用
  `stage_dependencies(stage_type, scoped.cfg)`，把结果放进该阶段的 `Checkpoint`；
  完成快照与 checkpoint 保存这份声明的依赖，恢复时按同一声明校验。
- **PrefixCache**：`metadata.pkl` 的 `dependencies` 直接取快照的
  `config_dependencies`，`find` 仍用 `matches` 匹配。
- **GpuScheduler**：`_prefix_paths(run, stage)` 是所有上游 Stage 与当前 Stage 声明路径
  的并集；`_stage_group` 把配置投影到该前缀；`_stage_vectors` 用同一投影生成估计特征。
  所有路径在派发前可得，因此分组和特征不再依赖历史样本，也没有冷启动的全配置回退。

## 4. 与旧机制的关系

- 旧 checkpoint 若是早期运行时追踪产生的信封记录，仍可读取并按其中的依赖匹配；
  不匹配则按损坏回退。
- 旧快照缺少依赖记录时不再自动保守依赖完整配置，而是以当前声明为准。
- 迁移成本：不重写 `config_dependencies` 的 Stage 变成整配置依赖，复用率下降但结果
  安全；需要恢复前缀复用时应显式声明读取的路径。

## 5. 验证

- `tests/test_gpu_scheduling.py`：声明决定 Stage 分组（未声明参数不拆组）、共享前缀
  物化、跨 Stage 前缀并集、恢复与中断语义。
- `tests/test_parallel_estimation.py`：声明前缀决定 Stage 组数与 ETA。
- 文档示例：`examples/shared_prefix.py` 等全部改为声明式。
