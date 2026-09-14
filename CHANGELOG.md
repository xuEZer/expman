# Changelog

## Unreleased

### Fixed

- `nvidia-smi` 在 PATH 中找不到时回退到 WSL2 的 `/usr/lib/wsl/lib/nvidia-smi`，避免 WSL2 下按未配置 PATH 启动批次时显存查询全部失败。

- 尝试输出不再常驻父进程：AttemptResult 只保留状态、耗时、错误摘要和读取来源，`result.output` 按需从 worker 结果记录或最终阶段快照读取，内存占用不随已完成实验数增长，中断后 `batch.results` 仍可读取全部输出。读取契约不变，但直接构造 AttemptResult 时输出字段名由 `output` 改为 `_output`。

- 调度准入同时观察逐卡显存与主机内存：每个 tick 重新查询两者，低于门槛时置 `mem_block`（主机内存）或该卡 `gpu_block`（显存）并停掉最近启动的一个实验，查询超时或失败按主机内存不足处理；block 只在对应卡或全局有实验自然退出时清除，减载自身的退出不清除；显存查询超时默认 3 秒；内存裕量门槛默认 1%（`MEMORY_MARGIN = 0.01`，8 GiB 卡约 82 MiB、32 GiB 主机约 320 MB）；去掉每卡 5 秒的启动间隔，每卡每 tick 最多启动一个实验；swap 只观测；并发 ETA 在压力期间只保留已占用的槽位。

- GPU 任务派发使用后台评分和增量多样性距离，CLI 读取后台 ETA，减少全队列重算对并发启动的阻塞。

- Batch 累计运行时间及最近有效剩余时间区间持久化，resume 后连续显示，恢复实时调度观测后更新预测。

- 命名默认配置统一从项目根目录的 configs 加载；示例默认参数迁移至项目 configs/models。

- checkpoint 原子保存并恢复实际配置依赖，避免下游无关参数变化导致前缀重算；保留序列化读取及旧检查点的保守回退。

### Added

- GPU Batch 改为按顶层 Stage 派发：已完成前缀从快照恢复，Stage 通过 IPC 上报完成状态、耗时、cgroup 内存峰值、PyTorch 显存峰值、Recorder 事件和配置读取依赖。按滚动依赖特征分别估计每个 Stage 的耗时、主机内存和显存峰值，并优先尝试与既有样本距离更远的可行参数组合。

- GPU Stage 调度按同一 Stage 的近邻配置历史取有界经验分位数估计耗时、主机内存和显存峰值，避免稀疏特征回归产生无界外推；显存合约不超过物理卡容量。cgroup 或 CUDA 内存拒绝会把本次合约作为有限下界，提高后续尝试的资源分配并重试。显式零 PyTorch 显存样本不再预留显存，但 Stage 仍获得可见 GPU。

- 阶段累计诊断耗时随 checkpoint 和完成快照原子保存、随恢复进度回退；共享复用保留源耗时并单独记录恢复用时。

- 同 Batch 连续前缀自动复用，保守配置读取追踪、共享快照引用、state/RNG/数值指标恢复及 GPU worker 接入。配置 get 和成员存在性查询改为显式报错。

- 根部 seed（默认 0）初始化和 seed_everything()；Python、NumPy/PyTorch 的全局随机状态随阶段快照与 checkpoint 原子保存、重试及进程恢复。

- YAML `device` 驱动的多卡/同卡多进程执行，显存门槛准入、退出门控补位、直接 kill 及跨进程恢复。
- 按卡 CLI 运行数、独立尝试日志、Recorder 汇总、并发 SQLite 写入和基于当前并发规模的剩余时间区间。

- `Batch.run()` 默认实时显示 CLI 进度、已运行时间与剩余时间，支持刷新间隔、关闭显示及中断收尾。
- `Batch.estimate()` / `TimeEstimate`：整个实验的剩余时间预测区间，覆盖水平默认 0.8，显示 `DD:HH:MM～DD:HH:MM`。
- 自动配置特征、支持未完成观测的贝叶斯耗时模型、信息价值调度及估计设置恢复。

- `ctx.log_metrics()`：嵌套数值指标按完整路径事务写入 SQLite，同键覆盖、跨重试及恢复保留，每个 Batch 共用数据库。

- 位置标识的阶段结果/state 原子快照、两份 checkpoint 和阶段状态自动更新。
- `Batch.resume()`：跨进程恢复配置、运行身份、队列和尝试记录；中断不消耗失败重试机会。
- `Serializer` / `PickleSerializer`、`StorageError`、`RecoveryWarning` 及运行目录互斥锁。
- `Batch` 和 `Experiment`：配置驱动的顺序执行、每次尝试的状态隔离、默认一次队尾重试及结果汇总。
- `ctx.cfg` 完整配置、`ctx.attempt` 尝试编号，以及关联实验、Pipeline 和 Stage 的观测事件。
- `load_configs()`：YAML 实验集合加载、嵌套 `!choice` 展开、按 `name` 字段路径自动加载并合并默认参数。
- `ConfigError` 和 `MissingConfigWarning`，以及独立 Run 配置、双模型 YAML 示例与配置行为测试。
- 同步顺序执行的 Pipeline 和泛型 Stage 基类。
- RunContext，以及带父子执行标识的生命周期、指标和进度事件。
- 单调时钟计时、异常传播、取消状态记录和记录器故障隔离。
- Recorder 协议和内存记录器。
- 可运行示例、核心行为测试、Python 包配置及 GitHub Actions 检查。
- 架构、贡献和版本管理文档。
- 固定版本的 Ruff lint、格式检查、每次提交自动执行的 pre-commit hook 和 CI 检查。

### Changed

- 项目统一使用 uv 和提交的 `uv.lock` 管理唯一环境；NumPy、PyTorch、Ruff 与 pre-commit 都是默认依赖，安装、开发、CI 和完整验证均通过 `uv sync`、`uv run` 执行。

- 新实验按估时信息价值选择执行顺序，取消成员优先续跑、失败成员保持队尾重试，结果顺序保持配置展开顺序。

- `ctx.cfg` 深层只读，运行时可变数据使用 `ctx.state`。重试恢复已完成阶段和最新可用 checkpoint。
- Pipeline 接收 Stage 类，每次执行创建独立实例。迁移时将 `Pipeline([MyStage()])` 改为 `Pipeline([MyStage])`，阶段以无参构造函数初始化，并在 `process()` 中通过 `ctx.cfg` 读取参数。
