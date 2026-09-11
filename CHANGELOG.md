# Changelog

## Unreleased

### Fixed

- 调度准入同时观察主机内存：`/proc/meminfo` 的 `MemAvailable` 比例低于门槛时暂停所有卡的派发并按最近启动顺序减载，减载后直到有实验自然退出才补位；swap 只观测；并发 ETA 在压力期间只保留已占用的槽位。

- GPU 任务派发使用后台评分和增量多样性距离，CLI 读取后台 ETA，减少全队列重算对并发启动的阻塞。

- Batch 累计运行时间及最近有效剩余时间区间持久化，resume 后连续显示，恢复实时调度观测后更新预测。

- GPU 显存查询暂时失败时暂停派发并有限重试，成功查询重置计数，持续失败才中断调度。

- 命名默认配置统一从项目根目录的 configs 加载；示例默认参数迁移至项目 configs/models。

- checkpoint 原子保存并恢复实际配置依赖，避免下游无关参数变化导致前缀重算；保留序列化读取及旧检查点的保守回退。

### Added

- 阶段累计诊断耗时随 checkpoint 和完成快照原子保存、随恢复进度回退；共享复用保留源耗时并单独记录恢复用时。

- 同 Batch 连续前缀自动复用，保守配置读取追踪、共享快照引用、state/RNG/数值指标恢复及 GPU worker 接入。配置 get 和成员存在性查询改为显式报错。

- 根部 seed（默认 0）初始化和 seed_everything()；Python、可选 NumPy/PyTorch 的全局随机状态随阶段快照与 checkpoint 原子保存、重试及进程恢复。

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

- 新实验按估时信息价值选择执行顺序，取消成员优先续跑、失败成员保持队尾重试，结果顺序保持配置展开顺序。

- `ctx.cfg` 深层只读，运行时可变数据使用 `ctx.state`。重试恢复已完成阶段和最新可用 checkpoint。
- Pipeline 接收 Stage 类，每次执行创建独立实例。迁移时将 `Pipeline([MyStage()])` 改为 `Pipeline([MyStage])`，阶段以无参构造函数初始化，并在 `process()` 中通过 `ctx.cfg` 读取参数。
