# Changelog

## Unreleased

### Added

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

- `ctx.cfg` 深层只读，运行时可变数据使用 `ctx.state`。重试恢复已完成阶段和最新可用 checkpoint。
- Pipeline 接收 Stage 类，每次执行创建独立实例。迁移时将 `Pipeline([MyStage()])` 改为 `Pipeline([MyStage])`，阶段以无参构造函数初始化，并在 `process()` 中通过 `ctx.cfg` 读取参数。
