# Changelog

## Unreleased (0.1.1+fixed1g)

### Breaking changes

- `expman run` 省略模块名时，默认导入模块由 `pipeline` 改为 `src.pipeline`。例如，`expman run train baseline` 从实验项目的 `src/pipeline.py` 导入 `train`。使用根目录 `pipeline.py` 的项目可显式运行 `expman run pipeline:train baseline`。恢复批次时，清单中的完整模块路径优先，只有清单仅记录属性名时才使用内置默认模块。
- 实验 YAML 的目录从 `configs/` 改为 `configs/experiments/`。从实验项目根目录执行 `expman run train baseline` 会读取 `configs/experiments/baseline.yaml`，命令中只需写配置名。YAML 的 `defaults` 按字段路径从 `./configs/` 读取，各条 CLI 命令的项目设置只读取 `./pyproject.toml` 中的 `[tool.expman]`。
- 非空 `Pipeline` 在构造时检查最后一个 Stage 的 `process()` 定义。叶子须为同步普通方法，只允许自然结束、`return` 和 `return None`。该约束也适用于独立和嵌套的 Pipeline。批次中的标量结果可通过 `ctx.log_metrics()` 保存，模型等产物由实验代码显式写入。尝试和实验结果移除按需读取最终返回值的 `output` 属性。
- 移除 Python 的 `Batch(pipeline, cfg)` 和 `Batch.resume(...)` 接口。批次统一通过 `expman run train config` 创建并运行，通过 `expman resume ID` 恢复。缺少流程位置的批次无法通过 CLI 恢复。
- 批次默认父目录由 `./runs/` 改为 `./batches/`。CLI 各子命令统一使用 `--batches-dir` 指定父目录，移除 `run` 的 `-o` / `--output-dir` 和 `web` 的 `--root`。CLI 以单个配置名新建批次时使用 `<pipeline-attr>_<config-stem>_<YYYYMMDDHHMMSS>` 命名子目录，并打印批次 ID 和目录。时间戳采用本地时间，精确到秒，目录重名时命令报错退出。
- `status`、`resume` 和 `stop` 的位置参数改为所选父目录中的整数批次 ID。`status` 省略 ID 时显示总览，指定 ID 时显示单批次详情。两种视图默认每两秒刷新一次，按 Ctrl+C 退出查看。批次清单读取失败时，命令报错并退出。`resume` 和 `stop` 要求提供 ID。
- `run` 和 `resume` 默认在后台执行批次，移除 `--progress` 选项。启动成功后命令打印批次 ID 和目录，以 0 退出。启动失败时返回非零退出码。各 run 的最终状态可通过 `expman status ID` 或 Web 看板查看，后台调度器的输出写入批次目录下的 `batch.log`。
- 移除 `expman web --read-only` 和 Python 看板服务的 `allow_stop` 参数。所有看板均允许访问者请求停止运行中的批次，`GET /api/health` 的响应也移除了 `stop` 字段。
- 批次清单 `batch.pkl` 使用 v2 格式。旧批次如需续跑，请使用创建它的代码环境。用当前代码从头运行时，`expman run` 会创建按时间戳命名的新批次目录。
- `expman run module:attr config` 要求该属性是 `Pipeline` 实例。原先导出阶段列表或零参工厂的模块，可分别写成 `train = Pipeline(stages)` 或 `train = make_pipeline()`，再用 `expman run module:train config` 启动。
- CLI 和 Web 看板共用批次父目录下的 `.batches.sqlite3` 登记批次。新批次分配当前最小的空闲非负整数 ID，其余已登记批次保留原编号。从旧 JSON 登记表切换后，可用 `expman status --batches-dir <父目录>` 核对编号。

### Added

- `expman run train baseline ablation` 支持将多份实验 YAML 合为一个批次。各文件按输入顺序独立展开，解析默认值后相同配置只运行一次。批次目录名按首次出现的顺序列出所有不同的配置名，如 `train_baseline_ablation_20261124103000`。
- 批次详情页新增“自动绘图”选项卡。横轴可选配置字段、指标或已有记录的 step，纵轴可多选随横轴变化的指标。按配置取值筛选后，可查看单次训练的曲线，或比较多次训练的均值和样本标准差误差棒。
- 批次详情页新增“运行情况”和“实验结果”选项卡。结果表列出全部展开配置，按配置显示每个指标最大 step 对应的数值，在展示值新增或改变时按单变量对照更新列顺序，并合并相邻同值单元格。
- 运行参数集中在 `src/expman/settings.py`。各条 CLI 命令可以共用项目 `pyproject.toml` 的 `[tool.expman]` 配置，Python 调用者可以用 `expman.set(...)` 和 `expman.get(key)` 覆盖或读取当前进程的设置。支持设置批次目录、看板端口、刷新间隔、资源预算和 worker 环境。
- 新增 `expman delete ID`，可删除指定批次的目录及登记记录并释放 ID。命令支持 `--batches-dir`，也可清除已手动移走目录的批次记录。运行中的批次须先停止。
- 默认 pytest 和提交检查均要求真实 CUDA/MPS 测试通过。测试从可见、显存足够且未被其他测试预留的物理 GPU 中选择利用率最低的一张，已有计算任务的卡也可入选。
- 运行看板由独立进程 `expman web` 提供。实验进程定期把只读快照写入 `live.json`，看板通过 HTTP 和 SSE 展示运行状态、调度与资源观测。支持指定地址、端口和批次根目录，并为发现的批次登记稳定编号。
- worker 默认将受支持的线程池环境变量设为 4，未设置 `HF_ENDPOINT` 时使用 `https://hf-mirror.com`。shell 已导出任一受支持的线程池变量时，整组线程设置由 shell 决定。可用 `EXPMAN_THREADS`（`off` 或正整数）和 `EXPMAN_HF_MIRROR`（`off`、`cn` 或端点）调整默认值。框架写入的值记录在每次尝试的 `bootstrap.pkl` 中。无效的 `EXPMAN_THREADS` 会在创建批次目录前报错。
- GPU Batch 改为按顶层 Stage 派发。每个后续阶段从直接前驱的快照恢复输入和随机状态，Stage 通过 IPC 上报完成状态、耗时、cgroup 内存峰值、NVIDIA 客户端显存采样峰值、Recorder 事件和声明的配置依赖。按滚动依赖特征分别估计每个 Stage 的耗时、主机内存和显存峰值。
- `Stage.config_dependencies(cfg)`：与配置树同构的分层依赖声明，`True` 表示整棵子树、映射递归到子键（序列用整数下标）、缺省或 `False` 表示不依赖；接收只读 `cfg`，可按取值选择分支。默认返回 `True`，未声明的 Stage 保守依赖完整配置。
- 调度器按同阶段近邻配置的历史样本估计主机和显存峰值。因主机限额或 CUDA OOM 取消的尝试会按提高的预算重新排队。
- 阶段累计诊断耗时在写入 checkpoint 和提交阶段完成记录时原子保存、随恢复进度回退；共享复用保留源耗时并单独记录恢复用时。
- 同 Batch 连续前缀自动复用，显式分层配置依赖、共享快照引用、state/RNG/数值指标恢复及 GPU worker 接入。配置 get 和成员存在性查询改为显式报错。
- Batch 使用配置根部必填的 `seed` 初始化随机状态。Python、NumPy 和 PyTorch 的全局随机状态随阶段快照与 checkpoint 保存及恢复。
- YAML `device` 驱动的多卡/同卡多进程执行，显存门槛准入、退出门控补位、直接 kill 及跨进程恢复。
- 独立尝试日志、Recorder 汇总、并发 SQLite 写入和基于当前并发规模的剩余时间区间。
- `TimeEstimate` 表示整批实验的剩余时间预测区间，覆盖水平默认 0.8，显示 `DD:HH:MM～DD:HH:MM`。
- 耗时估计使用已完成阶段的样本，取消尝试提供耗时下界。批次恢复时沿用已保存的覆盖率设置。
- `ctx.log_metrics()`：嵌套数值指标按完整路径事务写入 SQLite，同键覆盖、跨重试及恢复保留，每个 Batch 共用数据库。
- 位置标识的阶段结果/state 原子快照、单份 checkpoint 和阶段状态自动更新。
- `expman resume ID` 跨进程恢复配置、运行身份、队列和尝试记录。中断不消耗失败重试机会。
- `Serializer` / `PickleSerializer`、`StorageError`、`RecoveryWarning` 及运行目录互斥锁。
- 实验批次支持按配置创建 run、隔离各次尝试的状态并汇总结果。普通失败的阶段默认可重试一次，重试任务重新入队。
- `ctx.cfg` 完整配置、`ctx.attempt` 尝试编号，以及关联实验、Pipeline 和 Stage 的观测事件。
- 实验 YAML 支持按 `grid`、`scan` 展开配置组合，并按 `defaults` 所在字段的路径加载和合并默认参数。
- `ConfigError` 和 `MissingConfigWarning`，以及独立 Run 配置、双模型 YAML 示例与配置行为测试。
- 同步顺序执行的 Pipeline 和泛型 Stage 基类。
- RunContext，以及带父子执行标识的生命周期、指标和进度事件。
- 单调时钟计时、异常传播、取消状态记录和记录器故障隔离。
- Recorder 协议和内存记录器。
- 可运行示例、核心行为测试、Python 包配置及 GitHub Actions 检查。
- 源码导读与贡献指南。
- 固定版本的 Ruff lint、格式检查、每次提交自动执行的 pre-commit hook 和 CI 检查。

### Changed

- 共享中间节点的所有使用者成功完成直接后继阶段后，调度器立即回收该节点的快照和元数据。叶子只发布复用所需的元数据，所有仍可能复用该节点的 run 完成后回收。恢复批次时按已保存的进度继续清理。
- 看板按 `web_interval` 自动刷新总览与批次详情。阶段卡按已完成、运行中、排队中的顺序显示计数。worker 的内存与显存分列展示实际用量和预留量，资源分配区显示整机 CPU 和每张 GPU 的利用率。
- 显存改用 MPS 单客户端硬上限，启动时核验 PID 与 GPU。worker 在现有主机内存采样线程中一并查询 NVIDIA 客户端显存，CUDA OOM 后按提高的预算重试。
- 实验配置语法整体替换：`!choice`、顶层 `sweep` 与中间过渡的顶层 `over` 全部移除，改由三个顶层控制键表达——**`combines`** 取一个映射列表，每项是一份稀疏覆盖、逐项各跑一个 Run（配对同级字段），**`grid`** 列出各轴并跑它们的乘积（笛卡尔积），**`scan`** 在自身内部声明参数（文件不必重复书写，写了也会被覆盖）、以每条轴列出的第一个取值为基准点、一次只动一条轴（单参数扫描）。三者共用同一条形状规则：**列表表示该节点取其中一个值，映射表示继续往下走**，并且**可以叠加**：嵌套关系由外向内固定为 `combines` → `grid` → `scan`，Run 集是三层的乘积、内层变化最快；原先 `scan` 的列表形式（整份方案）改由 `combines` 承担。`expand` 段、`baseline` 开关、`over.mode`、以及用 `name` 触发默认查找一并消失，且**不提供任何兼容层或迁移提示**：旧写法要么退化为普通数据字段，要么按 YAML 语法错误被拒。`name` 保持为普通数据字段，查找职责由 `defaults: <name>` 承担。
- 新增 CLI 命令 `run`、`resume`、`status` 和 `stop`。CLI 创建批次时保存流程定位，恢复时自动导入。
- 尚无峰值样本的阶段分别预留 1 GiB 主机内存和 1 GiB 显存。调度器按实际余量选择可并行启动的任务，后续尝试使用测得的峰值及限额反馈。
- 剩余时间区间改用核加权的 log 尺度模型：中心是配置距离加权下的几何均值，带宽取约第 `n/8` 近的配置距离，取代固定 8 个最近邻的经验顺序统计量，避免样本跨过邻域边界时估计跳变，也不再从 8 个样本里取极值当区间端点；距离按声明配置项等权平均，多水平类别项只算一个变量，不再因为展开成多列 one-hot 而压倒数值项。上下界由留一残差的 conformal 顺序统计量给出，位置取 `⌈p·(n+1)⌉`、两侧各分走一半缺失概率，残差在样本自身的配置处测量；区间是乘性的，流程模块中的 `expman.set(coverage=...)` 指定单阶段耗时区间的名义覆盖率。被中断的尝试只测得耗时的下界，只进入下界一侧。`TimeEstimate.__str__` 渲染区间两端。
- 调度 worker 经 `Experiment.run_stage()` 执行一个顶层阶段。直接调用 `Pipeline.run()` 或 `Stage.run()` 在调用者进程内执行。
- 主机与显存准入改为按裸的峰值估计计价，110% 的保守系数只保留在硬上限（cgroup 内存限额与 MPS 客户端显存上限）上；运行中的 worker 也只按自身估计计入尚未用到的部分，超过估计的实测峰值仍会被计入。余量因此是单个尝试被允许超出的范围，而不是装箱可以重复花费的容量。
- 主机与显存峰值模型的配置距离改用与耗时模型相同的实现：按声明的配置项等权平均、每个变量归一到 1 以内，取代原先对编码列求平方和的做法。原先数值项（展开成 1 列，最多贡献 4）与类别项（展开成多列，恒贡献 2）的权重是 2:1，现在两类变量权重相同，距离也不再被平方放大。
- 配置依赖改为用户显式声明的分层结构 `Stage.config_dependencies(cfg)`；移除运行时的配置读取追踪，以及用于发现依赖的完整 Pipeline 探查阶段。冷启动分组与估计特征改为直接使用声明路径。
- 项目使用 uv 锁定运行与开发依赖。PyYAML、NumPy、PyTorch 随包安装，pytest、Pyright、pre-commit 属于开发依赖组；Ruff 由提交钩子和 CI 单独安装。
- 移除基于配置差异的调度优先级。同一阶段内，重试任务优先进入候选，其余任务按就绪队列顺序选取。批次结果仍按配置展开顺序返回。
- `ctx.cfg` 深层只读。阶段使用 `ctx.checkpoint(position, state)` 保存进度和业务状态，在重试时通过 `ctx.checkpoint_state` 读取。
- Pipeline 接收 Stage 类，每次执行创建独立实例。迁移时将 `Pipeline([MyStage()])` 改为 `Pipeline([MyStage])`，阶段以无参构造函数初始化，并在 `process()` 中通过 `ctx.cfg` 读取参数。
- MPS 在系统临时目录中创建独立的通信目录，并遵守 TMPDIR 设置。

### Removed

- 看板移除暂停按钮、刷新间隔下拉框和调度门闸卡片。调度事件保留尝试信息、结果和原因，省略内存与显存读数。
- 移除批次结果分析、看板的“结果分析”页签、`GET /api/analysis/<id>` 接口及对应的 `min_numeric_points`、`min_groups` 设置。
- `expman._time_model.features()`：编码入口统一为 `encode()`，该薄封装已无调用方。

### Fixed

- 条件依赖按每个阶段声明的完整路径和值进行分组、共享快照和恢复断点，避免不同实验得到同一份错误结果。配置轴展开保持 YAML 别名独立，默认参数逐层合并后校验设备列表。
- 混合容量 GPU 按各卡显存安排任务，因 cgroup 硬上限终止的 worker 按核验到的限额重新估价。ETA 会在任务完成后清零剩余组数，并在资源采样失败时暂停使用空闲槽位。
- 看板按最近可用的批次记录显示运行状态、历史组数和内存预算，时间区间的舍入与 CLI 一致。
- 停止批次改用本批次的控制管道，避免进程号被系统复用后通知无关进程。
- 批次 ID 复用后，看板的停止请求会核对批次目录名，避免操作已替换的批次。
- 中断的阶段优先恢复自己的 checkpoint。
- 并发执行 `resume` 时，竞争失败的进程会保留运行者的控制通道。`resume` 取得执行锁后会核对清单修订标识，读取后又有进程更新过的清单会触发报错，避免覆盖已保存的进度。
- 普通失败才消耗阶段的重试预算。取消后仍可重试，重试前已记录的 worker 耗时只计入总时长一次。
- 耗时估计按批次配置的覆盖率计算排队和运行中阶段的区间。两次排程的完工时间按大小排列，避免区间上下界倒置。
- 自定义 Recorder 抛出普通异常时会记录日志，主进程仍会处理 worker 随后发来的完成消息。
- YAML 配置展开时，日期与同文本字符串分别生成实验。集合类型配置的依赖摘要在不同 worker 间保持一致。已有 checkpoint 若记录过集合依赖，首次恢复时可能因摘要变化而重新计算该阶段。
- 远程监控与调度修复：内存控制器按实际挂载与进程归属解析，兼容 cgroup v1/v2/hybrid；独立组计数不可用时回退进程 RSS/HWM，不再把读取失败当成零用量。快照分别记录请求的内存上限与已核验的有效内核上限。
- GPU 装箱评分改按固定物理显存容量归一化，保留已有负载和未兑现预留，避免后续任务反复偏向剩余显存更少的卡；显存和主机内存准入边界仍按实时余量判断。
- 并发时间线按实际开始和结束时间绘制 worker 尝试，包含运行中的尝试。缓存复用显示在调度事件中。进度按 Run、Stage 和 attempt 匹配，防止阶段切换或重试后显示旧进度。
- ETA 统一按 worker 墙钟时间估算。仍在运行的尝试可以使用同阶段新到的样本，并根据已运行时长更新剩余估计，超过先前上界时也能继续估时。某个阶段首次运行时，可从该次顶层执行上报的进度估速。整批排程包含运行中 run 的后续阶段及阶段先后约束。
- 展开去重移到 `defaults` 解析之后：展开产物不同但解析结果相同的 Run（例如一份显式参数与一份指向相同内容的默认配置）不再重复跑。
- 批次统一使用 GPU Stage 调度。`device` 必填且必须是非空 GPU 编号列表，根部 `seed` 也必须显式提供，不再隐式使用 `0`。
- `nvidia-smi` 在 PATH 中找不到时回退到 WSL2 的 `/usr/lib/wsl/lib/nvidia-smi`，避免 WSL2 下按未配置 PATH 启动批次时显存查询全部失败。
- 调度准入同时观察逐卡显存与主机内存：每个 tick 重新查询两者，查询超时（默认 3 秒）或失败按主机内存紧张处理并暂停派发；已发出的预留若无法兑现——主机余量连零字节都放不下，或某卡上未兑现的预留超过空闲显存——则撤销最近启动的一个尝试（带冷却；只有一个 worker 时只暂停派发、不撤销）。准入由主机余量与逐卡未兑现预留联合决定，压力消失后自然恢复；swap 只观测；并发 ETA 在压力期间只保留已占用的槽位。
- 调度主循环轮流从各阶段取候选，仅按剩余主机内存和各卡剩余显存为可行的装箱方案打分。批次计时线程更新并保存 ETA。
- Batch 持久化累计运行时间及当前剩余时间区间，区间未知时记录为空。resume 沿用累计时间，调度恢复后更新预测。
- checkpoint 原子保存并按声明的配置依赖恢复，避免下游无关参数变化导致前缀重算。
