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
```

使用 uv 时，可以通过 `uv venv` 创建环境，再执行 `uv pip install -e .`。

## 定义实验阶段

用户继承 `Stage[输入类型, 输出类型]` 并实现 `process()`：

```python
from expman import InMemoryRecorder, Pipeline, RunContext, Stage


class Scale(Stage[list[float], list[float]]):
    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def process(self, data, ctx):
        result = [value * self.factor for value in data]
        ctx.report_metric("count", len(result))
        return result


recorder = InMemoryRecorder()
ctx = RunContext(recorder=recorder)
pipeline = Pipeline([Scale(2), Scale(0.5)], name="scaling")

assert pipeline.run([1.0, 2.0], ctx) == [1.0, 2.0]
for event in recorder.events:
    print(event)
```

`run()` 是统一执行入口，负责监测；`process()` 是用户扩展点。自定义构造函数需要调用 `super().__init__()`，可以通过 `name=` 设置阶段名称。阶段之间直接传递返回对象，输入输出契约由用户定义。

## 运行与观测约定

- Pipeline 按顺序执行；空 Pipeline 原样返回输入，默认输入为 `None`。
- 阶段失败会记录异常和耗时，原异常继续抛出，后续阶段停止执行。
- 每次执行都产生独立 ID；同名阶段、重复运行和嵌套执行可以通过父子 ID 区分。
- Stage 实例可以持有状态，重复使用实例会保留状态；独立实验应创建自己的实例。
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
