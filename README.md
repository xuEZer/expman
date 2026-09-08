# expman

用可组合的 `Pipeline` 和 `Stage` 组织实验流程，并统一记录阶段耗时、执行状态、指标和进度。

开发版本：**0.1.0**（尚未发布）。Python ≥ 3.10；执行核心使用标准库，YAML 配置加载使用 PyYAML。

## 安装与运行

在项目目录中执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/basic_pipeline.py
python examples/load_experiments.py
python examples/batch_pipeline.py
python examples/checkpoint_pipeline.py
```

使用 uv 时，可以通过 `uv venv` 创建环境，再执行 `uv pip install -e .`。

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
- 阶段可以独立调用 `stage.run(data, ctx)`。省略 `ctx` 时，每次顶层调用创建新的运行上下文。
- `ctx.report_metric("loss", 0.1, step=10)` 记录有限实数指标；`ctx.report_progress(10, total=100, unit="batch")` 上报绝对进度。总量未知时省略 `total`。
- 默认记录器将事件保存在内存。长实验可实现 `Recorder.record(event)`，将事件写入文件或数据库。记录器的普通异常会通过 Python logging 报告，业务执行继续。

耗时采用单调时钟，表示主机端经过时间。执行期间的内部上报和子阶段监测开销包含在内；GPU 异步计算的精确计时需要用户在业务代码中建立同步边界。

## 配置实验集合

使用 YAML 自由定义配置字段，`!choice` 表示候选值，普通列表保留为完整的参数值：

```yaml
# experiment.yaml
seed: !choice [0, 1, 2]
models:
  forecasting:
    name: patchtst
    patch_len: !choice [8, 16]
    hidden_sizes: [128, 64, 32]
```

`name` 触发默认参数查找。例如上述节点会加载实验 YAML 所在目录下的 `configs/models/forecasting/patchtst.yaml`：

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
- 返回结果按配置顺序排列，每个结果保留全部尝试的状态、耗时和错误摘要；`output` 是最终成功尝试的返回值。事件可通过 `batch.recorder` 获取，支持传入自定义 Recorder。
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
python -m pip install -e '.[dev]'
pre-commit install --install-hooks
ruff check .
ruff format --check .
python -m unittest discover -s tests -v
```

每次 Git 提交前必须通过 Ruff lint 和格式检查，提交 hook 会自动执行这两项检查。

源码布局、接口契约和扩展方向见 [架构设计](DESIGN.md)；分支、提交和版本发布约定见 [贡献指南](CONTRIBUTING.md)；版本变更见 [CHANGELOG](CHANGELOG.md)。
