# Changelog

## Unreleased

### Changed

- 新增命令行入口 `expman`（`run` / `resume` / `status` / `stop`）：pipeline 以 `module:attr` 定位，支持 `Pipeline` 实例、Stage 类列表或零参工厂，创建时记入 manifest 供 `resume` 自动重建；`status` 只读批目录展示各 run 进度与剩余时间区间，`stop` 向运行进程发送 SIGTERM 优雅停止（PID 记录在批目录 `expman.pid`，退出码 143），恢复仍走既有 checkpoint 通道。`Batch` 新增内部 `_pipeline_spec` 参数；库 API 行为不变。
- CLI 细化：`run` 默认输出到 `runs/<pipeline名>_<config名>`（重名时追加 `_2`、`_3`……）且默认不显示进度（`--progress` 打开）；`--retries` 与 `--coverage` 从命令行移除，改为进程级全局设置 `expman.set(retries=..., coverage=...)`，在 pipeline 模块里调用即可，CLI 因导入该模块而生效；`runs/` 下每个 Batch 在 `runs/.batches.json` 登记从 0 开始的稳定整数 ID（按创建顺序），`stop` / `resume` / `status` 均可按 ID 快捷引用；`status` 不带参数时持续刷新 `runs/` 下全部 Batch 的总览（runs 数、运行/待定/完成/失败/停止计数、阶段进度、已用与剩余时间），直到 Ctrl+C。
- 冷启动由“整个 Batch 串行推进单个 run”改为**探测式**：无样本的 Stage 不再无上限地单独运行，而是以设备的 1/4（`devices.PROBE_SHARE`，不低于 1 GiB 默认值、也不低于本 Batch 已观测的最大主机峰值）作为计价与硬上限并行探测；能同时启动几个由装箱算术决定。探测命中上限时与普通上限相同：以抬高的下界重估并重试。`limits.memory_limit` 移除 `cold_start` 参数，首次尝试不再无上限运行。

- 剩余时间区间改用核加权的 log 尺度模型：中心是配置距离加权下的几何均值，带宽取约第 `n/8` 近的配置距离，取代固定 8 个最近邻的经验顺序统计量，避免样本跨过邻域边界时估计跳变，也不再从 8 个样本里取极值当区间端点；距离按声明配置项等权平均，多水平类别项只算一个变量，不再因为展开成多列 one-hot 而压倒数值项。上下界由留一残差的 conformal 顺序统计量给出，位置取 `⌈p·(n+1)⌉`、两侧各分走一半缺失概率，残差在样本自身的配置处测量；区间是乘性的，`estimate_coverage` 因此成为区间的名义覆盖率而不是单点估计的分位数。被中断的尝试只测得耗时的下界，只进入下界一侧。`TimeEstimate.__str__` 渲染区间两端。

- `Batch` 成为唯一执行入口：`Experiment` 转为内部组件，不再从 `expman` 导出；移除 `Experiment.run()` 及其附属的 `_release_output()`、`_timing_snapshot()` 与 `ResultOutput`，调度 worker 一律经 `run_stage()` 执行单个顶层阶段。全部示例统一经 `Batch` 执行并显式声明 `device` 与根部 `seed`。

- 主机与显存准入改为按裸的峰值估计计价，110% 的保守系数只保留在硬上限（cgroup `memory.max` 与 PyTorch 分配器上限）上；运行中的 worker 也只按自身估计计入尚未用到的部分，超过估计的实测峰值仍会被计入。余量因此是单个尝试被允许超出的范围，而不是装箱可以重复花费的容量。

- 主机与显存峰值模型的配置距离改用与耗时模型相同的实现：按声明的配置项等权平均、每个变量归一到 1 以内，取代原先对编码列求平方和的做法。原先数值项（展开成 1 列，最多贡献 4）与类别项（展开成多列，恒贡献 2）的权重是 2:1，现在两类变量权重相同，距离也不再被平方放大。

### Removed

- `expman._time_model.features()`：编码入口统一为 `encode()`，该薄封装已无调用方。

### Fixed

- `Batch` 统一使用 GPU Stage 调度；`device` 必填且必须是非空 GPU 编号列表，根部 `seed` 也必须显式提供，不再隐式使用 `0`。

- `nvidia-smi` 在 PATH 中找不到时回退到 WSL2 的 `/usr/lib/wsl/lib/nvidia-smi`，避免 WSL2 下按未配置 PATH 启动批次时显存查询全部失败。

- 尝试输出不再常驻父进程：AttemptResult 只保留状态、耗时、错误摘要和读取来源，`result.output` 按需从 worker 结果记录或最终阶段快照读取，内存占用不随已完成实验数增长，中断后 `batch.results` 仍可读取全部输出。读取契约不变，但直接构造 AttemptResult 时输出字段名由 `output` 改为 `_output`。

- 调度准入同时观察逐卡显存与主机内存：每个 tick 重新查询两者，低于门槛时置 `mem_block`（主机内存）或该卡 `gpu_block`（显存）并停掉最近启动的一个实验，查询超时或失败按主机内存不足处理；block 只在对应卡或全局有实验自然退出时清除，减载自身的退出不清除；显存查询超时默认 3 秒；内存裕量门槛默认 1%（`MEMORY_MARGIN = 0.01`，8 GiB 卡约 82 MiB、32 GiB 主机约 320 MB）；去掉每卡 5 秒的启动间隔，每卡每 tick 最多启动一个实验；swap 只观测；并发 ETA 在压力期间只保留已占用的槽位。

- GPU 任务派发使用后台评分和增量多样性距离，CLI 读取后台 ETA，减少全队列重算对并发启动的阻塞。

- Batch 累计运行时间及最近有效剩余时间区间持久化，resume 后连续显示，恢复实时调度观测后更新预测。

- 命名默认配置统一从项目根目录的 configs 加载；示例默认参数迁移至项目 configs/models。

- checkpoint 原子保存并按声明的配置依赖恢复，避免下游无关参数变化导致前缀重算。

### Added

- 运行看板：`run` / `resume` 新增 `--web` / `--web-port` / `--web-host`，在当前进程内附带一个只读看板（回环 HTTP + Server-Sent Events，前端零外网依赖），实时展示各 run 状态计数、Batch 累计与剩余时间区间、各顶层 Stage 的剩余组数与单组耗时区间、调度队列与准入闸门、最近一次装箱计划、逐卡与主机的「分配 vs 实际」内存/显存（实心＝实测占用、虚线＝分配、红线＝硬上限，读数带 `age`、陈旧转灰）以及调度器事件流。看板只读：仅在调度器锁下拷贝已发布状态，绝不训练模型或改动准入，快照构建失败降级为 `degraded` 负载继续推送；`--web` 启动的实例开放 `POST /api/stop`（先应答再优雅停止，退出码 143），默认只读时返回 403。URL 写入批目录 `.webui.json`，`expman status` 顺带打印。绑定默认仅回环，端口被占自动顺延 10 个。

- 框架静默配置工作进程环境，项目无需自行声明：每个 worker 的线程上限固定为 4（`OMP`/`MKL`/`OPENBLAS`/`NUMEXPR`/`VECLIB`/`BLIS`/`RAYON`/`NUMBA` 的 `*_NUM_THREADS` 一并设置），避免并行尝试合计抢占 CPU；`HF_ENDPOINT` 未设置时指向 `https://hf-mirror.com`。shell 已导出的值一律优先（导出任一 `*_NUM_THREADS` 即整组交还），`EXPMAN_THREADS`（`off` 或正整数）与 `EXPMAN_HF_MIRROR`（`off`/`cn`/端点）可覆盖默认；写入值随该次尝试记入 `bootstrap.pkl` 的 `environment` 字段，非法取值在构造 Batch 时即报错，早于输出目录创建。

- GPU Batch 改为按顶层 Stage 派发：已完成前缀从快照恢复，Stage 通过 IPC 上报完成状态、耗时、cgroup 内存峰值、PyTorch 显存峰值、Recorder 事件和声明的配置依赖。按滚动依赖特征分别估计每个 Stage 的耗时、主机内存和显存峰值，并优先尝试与既有样本距离更远的可行参数组合。

- `Stage.config_dependencies(cfg)`：与配置树同构的分层依赖声明，`True` 表示整棵子树、映射递归到子键（序列用整数下标）、缺省或 `False` 表示不依赖；接收只读 `cfg`，可按取值选择分支。默认返回 `True`，未声明的 Stage 保守依赖完整配置。

- `load_configs()` 支持顶层 `sweep` 段做参数敏感性扫描：实验文件自身的字段是主实验基线，`axes` 用点号路径声明要扫的轴，默认 `mode: ofat` 一次只动一个轴（其余轴固定为基线），`mode: grid` 改用轴值笛卡尔积，`include_baseline` 控制是否附带基线条目。轴路径必须已存在于实验文件，路径经过 `!choice` 或默认文件出现 `sweep` 都会报错。

- `load_configs()` 允许以根级 `!choice` 组织配对组合：文档根可以是一组完整配置映射，候选之间让同级字段联动（数据集 A 配模型 a、数据集 B 配模型 b），候选内部仍可使用嵌套 `!choice` 与 `name` 默认。候选非映射、或候选内出现 `sweep` 都会报错。

- GPU Stage 调度按同一 Stage 的近邻配置历史取有界经验分位数估计耗时、主机内存和显存峰值，避免稀疏特征回归产生无界外推；显存合约不超过物理卡容量。cgroup 或 CUDA 内存拒绝会把本次合约作为有限下界，提高后续尝试的资源分配并重试。显式零 PyTorch 显存样本不再预留显存，但 Stage 仍获得可见 GPU。

- 阶段累计诊断耗时随 checkpoint 和完成快照原子保存、随恢复进度回退；共享复用保留源耗时并单独记录恢复用时。

- 同 Batch 连续前缀自动复用，显式分层配置依赖、共享快照引用、state/RNG/数值指标恢复及 GPU worker 接入。配置 get 和成员存在性查询改为显式报错。

- 根部 seed（默认 0）初始化和 seed_everything()；Python、NumPy/PyTorch 的全局随机状态随阶段快照与 checkpoint 原子保存、重试及进程恢复。

- YAML `device` 驱动的多卡/同卡多进程执行，显存门槛准入、退出门控补位、直接 kill 及跨进程恢复。
- 按卡 CLI 运行数、独立尝试日志、Recorder 汇总、并发 SQLite 写入和基于当前并发规模的剩余时间区间。

- `Batch.run()` 默认实时显示 CLI 进度、已运行时间与剩余时间，支持刷新间隔、关闭显示及中断收尾。
- `Batch.estimate()` / `TimeEstimate`：整个实验的剩余时间预测区间，覆盖水平默认 0.8，显示 `DD:HH:MM～DD:HH:MM`。
- 自动配置特征、支持未完成观测的贝叶斯耗时模型、信息价值调度及估计设置恢复。

- `ctx.log_metrics()`：嵌套数值指标按完整路径事务写入 SQLite，同键覆盖、跨重试及恢复保留，每个 Batch 共用数据库。

- 位置标识的阶段结果/state 原子快照、单份 checkpoint 和阶段状态自动更新。
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

- 配置依赖改为用户显式声明的分层结构 `Stage.config_dependencies(cfg)`；移除运行时的配置读取追踪，以及用于发现依赖的完整 Pipeline 探查阶段。冷启动分组与估计特征改为直接使用声明路径。

- 项目统一使用 uv 和提交的 `uv.lock` 管理唯一环境；NumPy、PyTorch、Ruff 与 pre-commit 都是默认依赖，安装、开发、CI 和完整验证均通过 `uv sync`、`uv run` 执行。

- 新实验按估时信息价值选择执行顺序，取消成员优先续跑、失败成员保持队尾重试，结果顺序保持配置展开顺序。

- `ctx.cfg` 深层只读，运行时可变数据使用 `ctx.state`。重试恢复已完成阶段和最新可用 checkpoint。
- Pipeline 接收 Stage 类，每次执行创建独立实例。迁移时将 `Pipeline([MyStage()])` 改为 `Pipeline([MyStage])`，阶段以无参构造函数初始化，并在 `process()` 中通过 `ctx.cfg` 读取参数。
