# Changelog

## Unreleased

### Added

- `load_configs()`：YAML 实验集合加载、嵌套 `!choice` 展开、按 `name` 字段路径自动加载并合并默认参数。
- `ConfigError` 和 `MissingConfigWarning`，以及独立 Run 配置、双模型 YAML 示例与配置行为测试。
- 同步顺序执行的 Pipeline 和泛型 Stage 基类。
- RunContext，以及带父子执行标识的生命周期、指标和进度事件。
- 单调时钟计时、异常传播、取消状态记录和记录器故障隔离。
- Recorder 协议和内存记录器。
- 可运行示例、核心行为测试、Python 包配置及 GitHub Actions 检查。
- 架构、贡献和版本管理文档。
- 固定版本的 Ruff lint、格式检查、每次提交自动执行的 pre-commit hook 和 CI 检查。
