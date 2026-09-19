# expman

用可组合的 `Pipeline` 和 `Stage` 组织实验流程，并统一记录阶段耗时、执行状态、指标和进度。

开发版本：**0.1.0**（尚未发布）。Python ≥ 3.10；执行核心使用标准库，YAML 配置加载使用 PyYAML。

## 安装与运行

项目统一由 [uv](https://docs.astral.sh/uv/) 管理；锁定环境只包含运行所需的东西（NumPy、PyTorch、PyYAML）。先安装 uv，再在项目目录执行：

```bash
uv sync
uv run python examples/basic_pipeline.py
uv run python examples/load_experiments.py
uv run python examples/batch_pipeline.py
uv run python examples/checkpoint_pipeline.py
```

## 定义实验阶段

用户继承 `Stage[输入类型, 输出类型]` 并实现 `process()`：

```python
from expman import InMemoryRecorder, Pipeline, RunContext, Stage


class Scale(Stage[list[float], list[float]]):
    def process(self, data, ctx):
        result = [value * ctx.cfg["factor"] for value in data]
        ctx.report_metric("count", len(result))
        return result


recorder = InMemoryRecorder()
ctx = RunContext(recorder=recorder, cfg={"factor": 2})
pipeline = Pipeline([Scale], name="scaling")

assert pipeline.run([1.0, 2.0], ctx) == [2.0, 4.0]
for event in recorder.events:
    print(event)
```

`run()` 是统一执行入口，负责监测；`process()` 是用户扩展点。Pipeline 接收 Stage 类，并在每次执行时无参实例化。自定义构造函数需要调用 `super().__init__()`，可以通过 `name=` 设置阶段名称。阶段之间直接传递返回对象，输入输出契约由用户定义。

## 运行与观测约定

- Pipeline 按顺序执行；空 Pipeline 原样返回输入，默认输入为 `None`。
- 阶段失败会记录异常和耗时，原异常继续抛出，后续阶段停止执行。
- 每次执行都产生独立 ID；同名阶段、重复运行和嵌套执行可以通过父子 ID 区分。
- Stage 实例可以持有状态；需要执行的阶段会创建新实例，恢复时跳过已完成阶段的构造和处理。
- 重写 `config_dependencies(cfg)` 声明阶段结果依赖的配置位置（见「同 Batch 连续前缀复用」）；默认返回 `True`，保守依赖完整配置。
- 阶段可以独立调用 `stage.run(data, ctx)`。省略 `ctx` 时，每次顶层调用创建新的运行上下文。
- `ctx.report_metric("loss", 0.1, step=10)` 记录有限实数指标；`ctx.report_progress(10, total=100, unit="batch")` 上报绝对进度。总量未知时省略 `total`。
- 默认记录器将事件保存在内存。长实验可实现 `Recorder.record(event)`，将事件写入文件或数据库。记录器的普通异常会通过 Python logging 报告，业务执行继续。

耗时采用单调时钟，表示主机端经过时间。执行期间的内部上报和子阶段监测开销包含在内；GPU 异步计算的精确计时需要用户在业务代码中建立同步边界。

## 配置实验集合

使用 YAML 自由定义配置字段，`!choice` 表示候选值，普通列表保留为完整的参数值：

```yaml
# experiment.yaml
seed: !choice [0, 1, 2]
device: [0]
models:
  forecasting:
    name: patchtst
    patch_len: !choice [8, 16]
    hidden_sizes: [128, 64, 32]
```

`name` 触发项目级默认参数查找。例如上述节点会加载 `<项目根目录>/configs/models/forecasting/patchtst.yaml`：

```yaml
patch_len: 32
d_model: 128
```

```python
from expman import load_configs

configs = load_configs("experiment.yaml")
assert len(configs) == 6
assert configs[0]["models"]["forecasting"]["patch_len"] == 8
assert configs[0]["models"]["forecasting"]["d_model"] == 128
```

每个字典表示一次具体运行，用户可以据此构建各自的 Pipeline。没有 `!choice` 时返回只含一个字典的列表。

- 独立候选按笛卡尔积展开，顺序遵循 YAML 字段和候选值的书写顺序。候选内部的选择只参与该分支的展开。
- 先展开实验配置，再查找默认文件；参数与 `name` 并列。字典递归合并，实验配置优先；列表整体替换，显式 `null` 覆盖默认值。
- 默认文件只允许固定参数，出现 `!choice` 会抛出 `ConfigError`。解析错误、重复键、非映射顶层也会报错，并包含来源文件。
- 缺失默认文件发出 `MissingConfigWarning` 并保留显式参数继续；同一次加载对同一路径只警告一次。
- `name` 在所有层次都是查找约定字段，默认参数中新增的嵌套 `name` 也会按最终层次解析。列表中的节点使用字段层次查找，列表索引不加入目录路径。
- 各个 Run 的字典及嵌套对象互相独立。函数一次性生成全部配置，参数组合很多时需留意内存占用。

完整的双模型配置见 [examples/experiment.yaml](examples/experiment.yaml)，加载示例见 [examples/load_experiments.py](examples/load_experiments.py)。

## 执行实验集合

Pipeline 定义流程，Batch 加载配置并生成 Experiment。每个阶段通过 `ctx.cfg` 读取当前实验的完整配置：

```python
from expman import Batch, Pipeline, Stage


class ModelSummary(Stage):
    def process(self, data, ctx):
        return {
            "seed": ctx.cfg["seed"],
            "models": ctx.cfg["models"],
        }


pipeline = Pipeline([ModelSummary])
batch = Batch(pipeline, cfg="examples/experiment.yaml", max_retries=1)
print(batch.output_dir)
results = batch.run()

for result in results:
    print(result.run_id, result.status.value, result.output)
    for attempt in result.attempts:
        print(attempt.attempt, attempt.status.value, attempt.error_message)
```

- 首次执行初始数据为 `None`。通过 Batch 或 Experiment 执行时，每个阶段自动保存返回值和 `ctx.state`；重试跳过已完成阶段，恢复其输出和状态。
- 任意普通异常（包括阶段构造异常）会记录失败并放到队尾，默认额外重试一次。设置 `max_retries=0` 可关闭重试。
- 重试保留 `run_id`，`ctx.attempt` 从 1 递增。`ctx.cfg` 深层只读，运行过程中变化的数据放在可变字典 `ctx.state`。
- `KeyboardInterrupt`、`SystemExit` 等中断会停止 Batch 并继续抛出。调用方捕获后可读取 `batch.results`：已执行实验保留结果，未执行实验状态为 `pending`。
- 新 Stage 按参数异质性和资源装箱选择启动顺序；返回结果按配置顺序排列，每个结果保留全部尝试的状态、耗时和错误摘要；`output` 是最终成功尝试的返回值。事件可通过 `batch.recorder` 获取，支持传入自定义 Recorder。
- 尝试输出不常驻内存：Batch 只保留状态、耗时、错误摘要和输出的存放位置，`result.output` 每次访问都从最终阶段快照读取，因此内存占用不随已完成实验数增长，中断后 `batch.results` 仍可读取全部输出。每次访问返回的是重新读到的副本，修改它不会写回记录；需要反复使用同一份数据时请自行保存引用。
- Batch 对象执行一次；继续已有实验使用 `Batch.resume()`，重新开始使用新的 Batch。也可以传入一份具体配置字典，创建只有一个 Experiment 的集合。

隔离覆盖框架持有的实例和配置；用户的类变量、全局变量及文件等外部副作用仍需自行管理。失败后不保存异常对象或 traceback，并触发垃圾回收。恢复不会回滚外部写入，用户代码需要合理处理重复执行。

完整示例见 [examples/batch_pipeline.py](examples/batch_pipeline.py)。

## 阶段快照与 checkpoint

Stage ID 是 Pipeline 中从 0 开始的位置，通过 `ctx.stage_id` 读取。返回值和 `ctx.state` 一起原子保存，保存成功才标记阶段完成；不可序列化的返回值或 state 会使该阶段失败并进入重试流程。

阶段内部可随时保存当前 state；进入 `process()` 前会自动恢复最近可用的 checkpoint：

```python
class Train(Stage):
    def process(self, data, ctx):
        model, optimizer = create_model_and_optimizer(ctx.cfg)
        if "model" in ctx.state:
            model.load_state_dict(ctx.state["model"])
            optimizer.load_state_dict(ctx.state["optimizer"])

        for epoch in range(ctx.state.get("next_epoch", 0), ctx.cfg["epochs"]):
            train_one_epoch(model, optimizer, data)
            ctx.state["model"] = model.state_dict()
            ctx.state["optimizer"] = optimizer.state_dict()
            ctx.state["next_epoch"] = epoch + 1
            ctx.checkpoint.save(step=epoch + 1)

        return model.state_dict()
```

其中模型创建和训练函数由用户实现。checkpoint 保存整个 state 和进度，固定保留两份成功写入的文件；`ctx.checkpoint.step` 是最近恢复或保存的进度。最新文件损坏时 warning 并读取上一份，两份都不可用时恢复阶段入口状态。没有 checkpoint 的失败阶段也从入口重跑。

默认目录为 `runs/<batch-id>/`，可通过 `output_dir=` 指定一个尚不存在的目录。配置在 Experiment 创建时保存一次，运行中的映射和列表使用只读包装；读取、索引、遍历照常，需要运行时对象时使用 state。

```python
# 原进程中创建并运行
batch = Batch(pipeline, "experiment.yaml", output_dir="runs/my-batch")
batch.run()

# Ctrl+C 或进程退出后，在新进程中提供相同的流程定义
batch = Batch.resume(pipeline, "runs/my-batch")
results = batch.run()
```

恢复使用持久化配置、实验 ID、尝试历史和待执行队列，不依赖原 YAML。中断不消耗失败重试机会；突然退出的未结束尝试记录为 cancelled，无法获知的耗时记为 0。恢复检查阶段类、顺序及可获取的类源码摘要；外部依赖和动态代码的兼容性仍由用户保证。同一目录只允许一个进程执行。

默认保存器 `PickleSerializer` 支持可 pickle 的 Python 对象，**只加载可信的本地快照**，并保持兼容的 Python、依赖及设备环境。文件句柄、生成器等不保证可保存。可实现 `Serializer.dump/load` 并在创建和恢复 Batch 时传入同一种 `serializer`。阶段、checkpoint 和 Batch 清单保存失败会报错；可选 Recorder 的普通故障仍按原约定处理，默认指标事件只存在于当前进程内存中。

运行 `python examples/checkpoint_pipeline.py --interrupt-once` 可以演示中断，随后按输出的 `--resume` 命令继续。

## 开发

```bash
uv sync
uv run python -m unittest discover -s tests -v

# Ruff 和 pre-commit 是开发工具，不在包的依赖里；按需装到环境外或临时加入：
python -m pip install pre-commit ruff
pre-commit install --install-hooks
ruff check .
ruff format --check .
```

每次 Git 提交前必须通过 Ruff lint 和格式检查，提交 hook 会自动执行这两项检查。

源码布局、接口契约和扩展方向见 [架构设计](DESIGN.md)；分支、提交和版本发布约定见 [贡献指南](CONTRIBUTING.md)；版本变更见 [CHANGELOG](CHANGELOG.md)。

### 持久化数值指标

在由 `Batch` 或 `Experiment` 管理的阶段内，使用 `ctx.log_metrics()` 保存嵌套指标：

```python
ctx.log_metrics(
    {"train": {"mse": 0.1, "mae": 0.2}, "val": {"mse": 0.15}},
    step=epoch,
)
ctx.log_metrics({"test": {"mse": 0.08}})  # 汇总指标
```

Batch 内所有实验共用输出目录中的 `metrics.sqlite3`；独立 Experiment 使用自身输出目录。数据库在首次非空写入时创建。每次调用先校验整棵字典，再以一个事务写入，返回即已提交。写入失败会使阶段失败，并进入 Batch 的重试流程。

指标名为非空字符串，叶子为有限实数（不接受布尔值、NaN 和无穷值），统一保存为 SQLite REAL（双精度浮点数）。`step` 为非负的 64 位有符号整数或省略。空字典不产生记录。嵌套深度受 Python 递归限制。

唯一键为实验 ID、完整阶段位置路径、step 和完整指标路径。同键写入覆盖数值、尝试编号和 UTC 更新时间；本次未传入的指标保留。恢复时保留所有旧记录，包括 checkpoint 之后的指标，重新运行到相同位置时再覆盖。已完成阶段复用快照时不会重新记录指标。

可使用 SQLite 工具或 Python 查询 `metrics` 表：

```python
import sqlite3

with sqlite3.connect(batch.output_dir / "metrics.sqlite3") as db:
    rows = db.execute(
        "SELECT run_id, stage_path, step, metric_path, value, attempt, updated_at "
        "FROM metrics WHERE run_id = ? ORDER BY stage_path, step, metric_path",
        (batch.experiments[0].run_id,),
    ).fetchall()
```

`stage_path` 和 `metric_path` 保存为 JSON 数组文本，例如 `[0]` 和 `["train", "mse"]`，可用 `json.loads()` 解码。嵌套 Pipeline 的位置路径包含调用序号，避免阶段冲突。省略 step 的汇总指标在数据库中使用 `-1`。`report_metric()` 仍用于向 Recorder 上报观测事件；持久化数值使用 `log_metrics()`。

### 实验级剩余时间区间

```python
batch = Batch(pipeline, cfg="experiment.yaml", estimate_coverage=0.8)
print(batch.estimate())  # ??:??:??～??:??:??
results = batch.run()
print(batch.estimate())  # 00:00:00～00:00:00
```

`batch.run()` 默认每秒在 CLI 的标准错误流刷新进度与剩余时间，无需修改业务阶段：

```text
运行中 3/10 (30%) | 成功 3 失败 0 | 当前 7/10 | 已运行 00:00:35 | 剩余 00:01:20～00:02:10
```

已运行时间采用 `DD:HH:MM`，按本次 `run()` 调用的实际经过时间统计，不足一分钟显示为零；恢复执行时重新计时，不包含停机时间。进度按已结束的实验计数，包括最终失败；等待重试的实验不计入完成数。“当前”是配置展开后的实验编号。普通终端原地刷新，窄终端使用完整状态行；重定向到文件时输出无控制字符的状态行，并省略重复状态。运行结束或 Ctrl+C 中断时输出最终状态并换行。

可用 `batch.run(refresh_interval=0.5)` 设置刷新间隔（正有限秒数），或用 `batch.run(progress=False)` 关闭显示。显示异常不会触发实验重试。程序仍可通过 `batch.estimate()` 读取估计。完整示例：

```bash
python examples/estimate_time.py
```

显示格式为 `DD:HH:MM～DD:HH:MM`，只展示预计剩余时间区间。天数至少两位且不截断，下界向下、上界向上取整到分钟。覆盖水平默认 `0.8`，对应第 10～90 百分位；可在创建 Batch 时设置 `estimate_coverage`，也可单次调用 `batch.estimate(coverage=0.95)`。参数必须介于 0 和 1 之间。全部结束（包括最终失败）时剩余为零。

`Batch.estimate()` 估计尚未物化的顶层 Stage 组的剩余完工时间。每个 Stage 从同 Stage 的近邻配置历史取经验分位数耗时，再按当前 GPU 并发槽位装箱；没有成功耗时样本时显示问号。Stage 的主机内存和 PyTorch 显存峰值也采用近邻经验分位数。

新 Stage 候选优先处理重试，之后选择与同 Stage 已采样参数距离较远的组合。联合装箱以主机内存和各卡显存利用情况为主要评分，异质性用于资源效果接近时的排序。返回结果仍按 YAML 展开顺序排列。

`Batch.resume()` 从持久化 Stage 历史重建估计，并保留覆盖设置。`TimeEstimate` 提供秒数上下界、样本数、剩余 Stage 组数及覆盖设置；`calibrated=False` 表示预测尚未经过实际工作负载校准。外部 GPU 竞争和未来失败次数会影响准确度。

`device` 是整个 Batch 必填的设备列表，不展开为实验组合，也不允许使用 `!choice` 或空列表。框架不改写用户的模型构建逻辑。`ctx.cfg` 保留完整的原始设备列表。每个顶层 Stage 都在独立解释器中运行，并分配一张可见的 GPU，业务代码可统一使用 `cuda:0`；不使用 CUDA 的 Stage 不会因此产生显存占用。绑定通过启动环境中的 GPU UUID 设置，在导入用户 Stage 模块前生效。

GPU Batch 以**顶层 Stage**为调度单元，而不是以整个 Pipeline 为单元。子进程只实际执行被派发的一个 Stage；此前完成的顶层 Stage 从快照恢复。每次完成会把该 Stage 的耗时、cgroup 内存当前值/峰值、PyTorch 分配器显存当前值/峰值和声明的配置依赖通过私有 IPC 通知调度器；Recorder 事件也走同一通道，不再靠轮询 worker 目录中的结果或事件文件。

模型按 Stage 分开保存。特征是滚动累积的声明依赖：某 Stage 的上游 Stage 声明的路径与它自身声明的路径共同决定特征。因此前缀依赖相同表示上游输入相同。依赖在运行前即可计算，冷启动也直接按声明路径挑选彼此更远的组合；有样本后，参数距离既用于候选排序，也用于从同 Stage 的最近历史样本中取保守经验分位数。耗时、主机内存峰值和显存峰值不会向未采样参数空间作概率外推：结果受有限历史样本约束；cgroup 截断的主机峰值仅按固定系数抬高一次，显存合约还不会超过物理卡容量。模型只在收到新的完成样本时重建，tick 不会用正在运行的瞬时数据改变估计。

默认值在 [devices.py](src/expman/devices.py) 中：主机预留 `HOST_RESERVE_KB = 1048576`（1 GiB），未知 Stage 的主机和显存峰值均为 1 GiB。每个 tick（`POLL_INTERVAL = 1.0` 秒）读取全局主机余量与 `nvidia-smi` 的整卡空闲显存，并收取 worker 的 IPC 上报；随后从每个就绪 Stage 取有限的异质参数候选，以有界 beam search 选择主机内存和各卡显存的联合装箱计划。计划优先最小化归一化的剩余资源向量，参数距离只用于打破资源效果相近的选择；因此显存型和内存型 Stage 会共同填充资源，而不会形成全局的前序 Stage 屏障。启动前必须满足：全局可用主机内存减预留，能覆盖新 Stage 的 cgroup 合约和每个运行 worker 尚未用到的合约部分；GPU 也必须覆盖新 Stage 的显存合约及同卡 worker 尚未使用的 PyTorch 分配器合约部分。

每个 worker 的主机内存上限由 cgroup `memory.max` 强制为其估计峰值的 110%（保留最低启动值）。cgroup 拒绝申请或峰值接近上限时，该 Stage 以更高的下界重新估计并重试，不消耗普通失败重试次数。显存上限使用 PyTorch 的 `torch.cuda.set_per_process_memory_fraction`；不使用其分配器的运行时无法得到通用的进程级显存硬上限。成功样本均报告零 PyTorch 峰值的 Stage 仍会分配可见 GPU，但其后续任务不预留显存、也不设置 PyTorch 分配器上限。

不再有显存比例门槛或 CUDA OOM 封卡。无论是 PyTorch 分配器上限还是外部竞争导致的 CUDA OOM，都会作为该 Stage 超出当前显存合约的证据：调度器以本次合约为有限下界提高下一次分配并重试。每个 tick 仍根据最新整卡空闲显存和运行 worker 尚未使用的合约部分决定可行派发；主机内存读数失败或持续紧张仍暂停新增派发并按原有规则减载最新 worker。

Ctrl+C 立即停止调度并 kill 所有实验进程组，不要求子进程额外保存。主进程记录已知尝试耗时和中断队列；恢复复用阶段快照及最近有效 checkpoint。主进程突然结束时，子进程通过父进程管道 EOF 清理自己的进程组；未落盘的尝试耗时继续按未知处理。用户自行脱离实验进程组的外部服务不在管理范围内。

Batch 清单只由主进程更新。指标仍同步写入同一个 SQLite 数据库，跨进程写入等待锁最长 30 秒。Recorder 事件通过 IPC 转交主进程，单进程事件保持顺序，不承诺跨进程的全局发生顺序；强制退出时尚未发出的事件可能丢失。子进程的 stdout/stderr 位于 `experiments/<run_id>/attempts/stage-<index>/<attempt>/output.log`，CLI 由主进程统一输出各卡运行数量、整体进度及时间。

GPU 执行目前支持 Linux/WSL，需要可查询选定设备显存的 NVIDIA 驱动、`nvidia-smi` 和可读的 `/proc/meminfo`；`nvidia-smi` 先从 `PATH` 查找，找不到时回退到 WSL2 的 `/usr/lib/wsl/lib/nvidia-smi`（WSL2 默认不把它加入 `PATH`）。任一路径查询失败都按资源紧张处理，暂停派发并减载，不会中断本批次。Stage 应定义为可导入模块或入口脚本的顶层类，入口使用 `if __name__ == "__main__":`；Pipeline 和自定义 Serializer 需要可通过标准 pickle 传入新解释器。子进程 stdin 用于检测主进程退出，不支持交互式输入。保存器的对象/设备恢复语义保持原约定；建议把需要在主进程汇总的结果转换为 CPU 对象，避免反序列化结果时在主进程占用 GPU。

并发 ETA 使用真实实验耗时，并增加运行设备和同卡并发观测；联合抽样后模拟各卡当前并发槽位的完成时间，取整批最晚完成时间的区间。它是以当前并发规模为条件的预测，不用总时间除以 GPU 数，也不承诺进程公平分享算力。主机内存低于门槛时，空闲卡不再获得预测槽位，显示的是不再扩大并发的剩余时间；没有实验在运行且主机内存不足时剩余时间未知。未来显存变化引起的启停、外部竞争及硬件差异仍会影响准确度，`calibrated=False`；没有成功样本或无可用卡且无运行实验时显示问号。设备和并发历史保存在 Batch 清单中以供恢复。

示例包含 checkpoint 和数值指标；默认 YAML 使用 CPU，可将 `device` 改成实际 GPU 列表再运行：

```bash
python examples/multi_gpu.py
python examples/multi_gpu.py --resume runs/<batch_id>
```

设备 UUID 与可见设备编号的规则参见 [NVIDIA CUDA_VISIBLE_DEVICES 文档](https://docs.nvidia.com/deploy/topics/topic_5_2_1.html)。


### 随机种子与随机状态恢复

Experiment 启动时读取必填的根部 `seed`，并在构造和执行 Stage 前设置 Python `random`、环境中已安装的 NumPy 全局随机生成器，以及 PyTorch CPU/CUDA 随机种子。种子必须是 `0` 到 `2**32 - 1` 的整数，不接受布尔值。嵌套配置中的同名字段由用户代码解释。

```yaml
seed: 42
device: [0]
models:
  forecasting:
    name: patchtst
```

NumPy/PyTorch 包含在 uv 锁定环境中；run 启动时主动导入它们，不需要用户预先 import。手动删改环境导致导入失败时会正常报错。可以在库外单独调用相同的初始化函数：

```python
from expman import seed_everything

seed_everything(42)
```

每个 run 将初始化后的随机状态原子写入 `experiments/<run_id>/rng_initial.pkl`。重试和恢复读取此记录，不重复设种子；随后按已有阶段快照和 checkpoint 恢复到相应位置。阶段完成快照及 checkpoint 都包含独立的 `rng_state` 字段，与业务 state 在同一文件、同一事务中保存，不占用 `ctx.state`。框架仍只保留最近两份 checkpoint。

复用本 run 已完成阶段时恢复该阶段结束时的随机状态；失败阶段有 checkpoint 时恢复它，没有则从阶段入口的随机状态重跑。为避免 Stage 构造函数消耗随机数影响续跑，checkpoint 的随机状态在构造前验证恢复，并在构造后、执行阶段前再次恢复。用户在 `process()` 中重建模型、恢复数据迭代器等准备工作若消耗随机数，仍需自行管理这段恢复逻辑。

随机状态记录覆盖 Python 全局生成器（含 Gaussian 缓存）、NumPy 全局 RandomState、PyTorch CPU 及全部可见 CUDA 设备的生成器。NumPy 数组转为普通列表、PyTorch ByteTensor 转为 bytes 存储，因此仅查看快照元数据不会为了随机状态反序列化 CUDA Tensor。保存可见 CUDA 状态会初始化相关生成器，需要可用的 CUDA 环境；GPU 调度器已在子进程启动前限制设备可见范围。

损坏的随机 checkpoint 会 warning 并尝试上一份，损坏的初始随机状态或已完成阶段随机状态会报错。旧版快照缺少随机状态时仍可恢复业务数据，但会发出 `RecoveryWarning`，不能保证随机序列连续。恢复所需的随机库缺失、CUDA 状态数量与可见设备不一致等不兼容情况会报错。

独立的 `random.Random`、NumPy `Generator/default_rng`、`torch.Generator`、DataLoader 工作进程及第三方库的状态由用户保存。此功能不自动开启确定性 GPU 算法、不控制 Python hash 随机化，也不保证跨库版本或硬件逐位一致；参见 [PyTorch 可复现性说明](https://docs.pytorch.org/docs/stable/notes/randomness.html)。直接使用未受 Experiment 管理的 Pipeline/Stage，不会自动设置全局种子。

完整示例：`python examples/random_state.py`。

## 同 Batch 连续前缀复用

Batch 自动共享已完成的阶段结果。Pipeline 从 0 号阶段开始匹配声明的配置依赖；一旦某阶段不匹配，本次 run 后续阶段全部实际执行。根部 seed 和 Pipeline 定义参与缓存隔离。每个 Stage 通过 `config_dependencies(cfg)` 显式声明依赖，框架只按声明匹配缓存：不再运行探查阶段，也不在运行时记录配置读取。

例如两个实验仅 model 不同，阶段 0 声明只依赖 data，阶段 1 声明只依赖 model，那么第二个实验可以复用阶段 0。命中时恢复返回值、完整 state 和随机数状态，并将该阶段的数值指标写入当前 run 的 SQLite 记录。示例：`python examples/shared_prefix.py`。

依赖声明与配置树同构：节点为 `True` 表示依赖该子树，为映射则递归到子键（序列用整数下标），缺省或 `False` 表示不依赖，返回 `True` 表示依赖完整配置。声明方法接收只读的 `cfg`，因此可以按取值选择分支——例如 `Baseline` 为 `B1` 时只声明 `imputer`，为 `B2` 时同时声明 `predictor`。默认返回 `True`：读取 `ctx.cfg` 却未重写该方法的 Stage 会保守地依赖完整配置，不会错误复用。配置中的字段必须显式声明，`get()` 和 `in` 判断抛出 TypeError，索引不存在的字段抛出 KeyError。state 仍是普通字典。checkpoint 和完成快照保存当前声明的依赖，恢复时按同一声明校验。

共享快照位于 Batch 的 cache 目录，各 run 的 completed.pkl 保存引用；移动实验记录时应保留完整 Batch 目录。自身已有快照和 checkpoint 优先恢复。并发进程只读取已经发布完成的共享节点，同时启动的相同工作仍可能各自计算。嵌套 Pipeline 的依赖由外层 Stage 的声明覆盖。

复用以相同上游、相关配置和随机状态产生一致结果为前提；文件内容变化等外部输入需通过配置中的版本字段表达。跳过阶段不会重新执行其中的外部副作用。独立 Experiment 保留自身恢复行为，共享范围限于同一个 Batch。

## 可恢复的阶段诊断计时

每个阶段的完成快照及 checkpoint 都包含 `elapsed_seconds`（秒）。阶段恢复 state 和随机状态之后、构造 Stage 之前启动单调时钟；保存时将恢复的累计值与本次执行时长相加，与进度写入同一原子记录。计时字段由框架管理，不占用 ctx.state。

例如 checkpoint 记录 100 秒，随后运行 20 秒后中断；恢复后再运行 30 秒完成，阶段快照记录 130 秒。没有 checkpoint 的失败阶段重跑时从零计时。停机时间和恢复检查点的加载时间不计入累计值。保存时取写入前的时间截点，因此当前保存操作的耗时不在该记录内；不中断继续运行时，它会进入下一次记录的执行时长。嵌套阶段独立计时，外层耗时包含内层执行，不能直接将各层耗时相加。

共享复用保留源快照的 elapsed_seconds，本 run 的 status.pkl 另存 restore_seconds，覆盖查找、加载及恢复操作（截至写入状态之前）。正常完成的 status.pkl 同步记录阶段累计耗时。旧快照缺少计时字段时视为未知，续跑后的累计值保持 None，避免把部分时长当作完整历史。损坏的 checkpoint 计时字段触发与 state 相同的回退流程。

Batch 的尝试耗时与剩余时间估计保持原有规则，仍计入真实的失败、中断执行消耗。阶段计时用于实验诊断。

配置根目录统一为 `<项目根目录>/configs/`，`data.name: demo` 加载其中的 `data/demo.yaml`。项目根目录通过向上查找 `pyproject.toml` 或 `.git` 定位：先从实验 YAML 所在目录查找，未找到项目标识时从当前工作目录查找。均未找到时，加载命名配置会报 ConfigError。实验 YAML 可放在项目的 `configs/`、其子目录或其他目录。示例默认参数位于项目根部的 `configs/models/`。

GPU 显存查询使用内部默认超时 3 秒，每个调度 tick 重新查询显存与主机内存。该查询实测在 WSL2 单卡机器上中位约 53 毫秒、最坏约 0.9 秒（宿主机 CPU 满载时），读取 `/proc/meminfo` 不足 1 毫秒，因此单次 tick 的查询开销约为 1 秒轮询间隔的百分之五到九成。查询失败、超时或返回无效数据时清除旧读数、发出 `RuntimeWarning`、置 `mem_block` 并减载一次，下一个 tick 重新查询，调度不因查询失败终止。GPU 身份改变属于不可恢复的设备一致性错误。上述参数由框架内部定义。

Batch 的累计墙钟时间与最近有效剩余时间区间原子保存在 `timing.pkl`。运行期间约每 5 秒更新一次，正常结束或捕获中断时再保存；关闭 CLI 仍记录。`Batch.elapsed_seconds` 返回跨 resume 累计运行时间，排除两次运行之间的停机时间，并行实验不重复累计。突然退出只能恢复最近成功保存的累计值。GPU 恢复初期尚无可用调度数据时，直接显示保存的同覆盖率预测区间，取得有效数据后更新。旧目录缺少 timing.pkl 时无法还原历史墙钟时间，从零开始累计；原有实验耗时及并发历史仍用于重新估计。

GPU 调度先为每个就绪 Stage 生成重试优先、参数距离靠前的候选，再联合主机和每张卡的容量选择一组可同时启动的任务。模型只在新的 Stage 完成 IPC 到达时更新；CLI 仍读取最近有效的 Batch ETA，后台计时线程统一刷新默认覆盖率的预测并保存。并行 ETA 的历史记录按 run_id 索引，避免每个实验扫描整份历史。
