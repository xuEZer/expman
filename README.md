# expman

用可组合的 `Pipeline` 和 `Stage` 组织实验流程，并统一记录阶段耗时、执行状态、指标和进度。

需要 Python ≥ 3.12、PyYAML、NumPy 和 PyTorch。

## 执行模型

从一次实验看，`Stage.process` 完成一步计算，`Pipeline` 定义这些步骤的顺序。一份具体配置对应一个 `Experiment`。直接调用 `Pipeline.run` 可以在当前进程运行这条流程。

批量运行从 `Batch.run` 或 `expman run` 进入。调度器把每个需要计算的顶层 Stage 派发到独立工作进程，并负责 GPU 分配、资源限制、重试与恢复。匹配的已完成前缀可直接复用。运行环境要求 **Linux 与支持 MPS 的 NVIDIA GPU**（需要 `nvidia-smi`、`nvidia-cuda-mps-control` 和 CUDA 驱动）；每份实验配置必须显式提供 `device`（GPU 编号列表）和根部 `seed`。各配置使用同一份物理 GPU 列表。

接手源码可从批次创建、派发一个阶段、保存结果这三步读起。[架构与源码阅读顺序](DESIGN.md)

## 安装与运行

项目使用 uv 管理环境。运行依赖为 PyYAML、NumPy 和 PyTorch，默认同步时也会安装开发依赖组。在仓库根目录执行

```bash
uv sync
source .venv/bin/activate
```

运行与恢复见《命令行》一节：`expman run <module:attr> <config>`。

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

`process()` 接收上一步的返回值和当前上下文。直接调用 Stage 时用 `stage.run(data, ctx)` 记录执行事件。Pipeline 接收 Stage 类，需要计算某一步时才无参构造实例。自定义构造函数应调用 `super().__init__()`，可通过 `name=` 指定阶段名称。

## 运行与观测约定

- Pipeline 按顺序执行；空 Pipeline 原样返回输入，默认输入为 `None`。
- 直接调用时，阶段抛出的异常会向调用方传播。Batch 中的 worker 会将业务失败报告给调度器，由调度器决定是否重试。
- 每次执行都产生独立 ID；同名阶段、重复运行和嵌套执行可以通过父子 ID 区分。
- Stage 实例可以持有状态；需要执行的阶段会创建新实例，恢复时跳过已完成阶段的构造和处理。
- 重写 `config_dependencies(cfg)` 声明阶段结果依赖的配置位置（见「同 Batch 连续前缀复用」）；默认返回 `True`，保守依赖完整配置。
- 阶段可以独立调用 `stage.run(data, ctx)`。省略 `ctx` 时，每次顶层调用创建新的运行上下文。
- `ctx.report_metric("loss", 0.1, step=10)` 记录有限实数指标；`ctx.report_progress(10, total=100, unit="batch")` 上报绝对进度。总量未知时省略 `total`。
- 默认记录器将事件保存在内存。长实验可实现 `Recorder.record(event)`，将事件写入文件或数据库。记录器的普通异常会通过 Python logging 报告，业务执行继续。

耗时采用单调时钟，表示主机端经过时间。执行期间的内部上报和子阶段监测开销包含在内；GPU 异步计算的精确计时需要用户在业务代码中建立同步边界。

## 配置实验集合

使用 YAML 自由定义配置字段。普通列表始终是一个完整的参数值；需要一次跑多组配置时，加顶层控制键。**形状规则只有一条**——列表表示该节点取其中一个值，映射表示继续往下走——而**跑法由关键字决定**：

| 控制键 | 取值 | 跑法 |
| --- | --- | --- |
| `combines` | 映射的列表 | 整份方案：每项是一份稀疏覆盖，逐项各跑一个 Run，项与项之间不组合 |
| `grid` | 映射 | 笛卡尔积：列出的轴同时变，Run 集是各轴取值的乘积 |
| `scan` | 映射 | 一次只动一个：每条轴取自己列出的第一个值当基准，其余取值逐个偏离 |

三者是**三层**，由外向内 `combines` → `grid` → `scan`，写几个就乘几层（见「叠加三层控制键」）。

### 笛卡尔积：`grid`

```yaml
device: [0]
seed: 0

models:
  imputation: {name: saits, hidden_size: 128}
  forecasting: {name: patchtst, patch_len: 8, d_model: 128}

grid:                          # seed × imputation × patch_len，2×2×2 = 8 个 Run
  seed: [0, 1]
  models:
    imputation:
      - {name: saits, hidden_size: 128}
      - {name: brits}
    forecasting:
      patch_len: [8, 16]
```

```python
from expman import load_configs

configs = load_configs("experiment.yaml")
assert len(configs) == 8
assert configs[0]["models"]["imputation"]["name"] == "saits"
assert configs[0]["models"]["forecasting"]["d_model"] == 128
```

- 列出的字段由 `grid` 接管，文件里的取值被替换：上例 `seed` 写的是 0，实际跑的是 0 和 1。
- 展开顺序由 `grid` 的书写顺序决定，与数据字段的书写顺序无关；多轴时最后一个轴变化最快。
- 轴既可以落在叶子上，也可以落在子树上。子树候选整体替换该节点，仍然与其它轴相乘。
- 每个叶子都必须是列表；标量叶子没有意义，会报错。

### 单参数扫描：`scan`

`scan` 里的参数**只写在 `scan` 里**，文件不必重复定义。每条轴拿自己列出的**第一个值**当基准，其余取值逐个偏离，其它轴留在基准值：

```yaml
stage: B2
device: [0]
seed: 42
tip:
  cone_delta: 0.03        # 不扫描的字段留在文件里

scan:
  tip:
    subspace_rank: [2, 4]      # 轴一
    epsilon: [0.00001, 0.001]  # 轴二
```

上例 3 个 Run：基准点（rank=2 / epsilon=0.00001 / cone=0.03）打头，然后两条轴各偏离一次——动 `subspace_rank` 时 `epsilon` 停在基准的 0.00001，反之亦然。要一次跑满 2×2 = 4，改用 `grid`。

```python
from expman import load_configs

configs = load_configs("experiment.yaml")
assert len(configs) == 3
assert [cfg["tip"] for cfg in configs] == [
    {"subspace_rank": 2, "epsilon": 0.00001, "cone_delta": 0.03},
    {"subspace_rank": 4, "epsilon": 0.00001, "cone_delta": 0.03},
    {"subspace_rank": 2, "epsilon": 0.001, "cone_delta": 0.03},
]
```

- 基准点永远是第一个 Run，也永远产出，没有开关可以去掉。
- 基准值是「第一个列出的值」，**不是文件里的值**：同一个字段在文件里写了也会被 `scan` 覆盖。想换基准就把值挪到列表首位，写成 `[3, 2, 4]`。
- 一条轴列 N 个取值 → 1 个基准值 + N−1 个偏离 Run；重复取值不会重复跑。
- 轴指向的字段不必预先存在，`scan` 自己声明它。

### 整份方案：`combines`

要让同级字段联动（数据集 A 配模型 a、数据集 B 配模型 b），把每套配置写成 `combines` 上的一项——**列表放在 `combines` 本身的位置**，各项之间不组合，各自作为一个 Run：

```yaml
# experiment.yaml：每个数据集各自绑定一个模型，而不是 2×2×2 的全组合
device: [0]
seed: 42

combines:
  - dataset: {name: azure2019_I_5T}
    imputer: {name: SAITS}
    predictor: {name: Toto2}
  - dataset: {name: EWELD_Load_15T}
    imputer: {name: DeepMVI}
    predictor: {name: Chronos}
```

```python
from expman import load_configs

configs = load_configs("experiment.yaml")
assert len(configs) == 2
assert [cfg["dataset"]["name"] for cfg in configs] == [
    "azure2019_I_5T",
    "EWELD_Load_15T",
]
```

- 上例 2 个 Run，`dataset` / `imputer` / `predictor` 按项配对，而不是 2×2×2。
- 每项是**稀疏覆盖**：只覆盖它列出的字段，映射也只往下覆盖，其余字段（上例的 `device`、`seed`）沿用文件。
- 一项可以只写一格字段：`- {seed: 7}` 就是「其余照文件、只把 seed 换成 7」。
- `combines` 只接受列表，项必须是映射；项里的列表是普通数据，不当作轴。

### 叠加三层控制键

三个键可以同时写，嵌套关系由外向内固定为 `combines` → `grid` → `scan`；`grid` 与 `scan` 都作用在上一个键产出之后的文档上：

```yaml
stage: B1
device: [0]
seed: 42

combines:
  - {stage: A}
  - {stage: B}

grid:
  seed: [0, 1]

scan:
  rank: [2, 4]
```

```python
from expman import load_configs

configs = load_configs("experiment.yaml")
assert len(configs) == 8
assert [(cfg["stage"], cfg["seed"], cfg["rank"]) for cfg in configs[:4]] == [
    ("A", 0, 2),
    ("A", 0, 4),
    ("A", 1, 2),
    ("A", 1, 4),
]
```

上例是 2 × 2 × 2 = **8 个 Run**：外层 `combines` 最慢，内层 `scan` 最快。`grid` 的「键必须已存在」检查在它那一层做，所以 `combines` 新引入的字段可以成为 `grid` 的轴。

### 共同规则

- 在实验 YAML 的根节点，`combines`、`grid` 和 `scan` 描述展开方式。这三个根键在展开前取出。嵌套映射中的同名键是普通数据。`defaults` 在解析默认参数时被取出。
- `grid` 的稀疏副本里的键必须已存在于文档中（默认值先合并、再展开），所以覆盖 `defaults` 给出的参数是允许的；`combines` 的项与 `scan` 自己声明参数，路径不必预先存在。
- `device` 是 Batch 级的资源列表，不允许作为展开轴。
- 展开后按结构去重，重复的配置不会跑第二遍；去重在 `defaults` 解析之后进行，跨层生效。
- `grid` 和 `scan` 接受非空的轴映射，每条轴列出至少一个候选值。`combines` 接受非空列表，每项是映射。候选值本身可以是空列表、空映射或 `null`。省略三个控制键时只生成一份配置。

### 命名默认参数

`defaults` 可以写在任何映射节点上，按字段路径查找项目级默认文件：

```text
models.forecasting.defaults: patchtst
  → <项目根目录>/configs/models/forecasting/patchtst.yaml
```

```yaml
# configs/models/forecasting/patchtst.yaml
name: patchtst
patch_len: 32
d_model: 128
```

- 参数与 `defaults` 并列书写。两侧都是字典时递归合并，实验配置优先；列表整体替换，显式 `null` 覆盖默认值。
- `defaults` 只是查找指令，不会出现在 Run 配置里；`name` 因此是普通数据字段，模型身份由默认文件自己给出。
- 根节点的 `defaults` 对应 `configs/<name>.yaml`；列表元素使用其所在字段路径，列表索引不进入目录路径。
- 默认文件的根节点出现 `combines`、`grid` 或 `scan` 会抛出 `ConfigError`。
- 缺失默认文件发出 `MissingConfigWarning` 并保留显式参数继续；同一次加载对同一路径只警告一次。
- 各个 Run 的字典及嵌套对象互相独立；加载器一次性生成全部配置，参数组合很多时需留意内存占用。

## 执行实验集合

将上面含有 `models` 的 `grid` 示例保存为 `experiment.yaml`。下面的脚本定义一个 Stage，并在主进程中启动 Batch。worker 会重新加载这个脚本以查找 Stage 类，因此启动代码放在 `if __name__ == "__main__":` 中。

```python
from expman import Batch, Pipeline, Stage


class ModelSummary(Stage):
    def process(self, data, ctx):
        return {
            "seed": ctx.cfg["seed"],
            "models": ctx.cfg["models"],
        }


if __name__ == "__main__":
    pipeline = Pipeline([ModelSummary])
    batch = Batch(pipeline, cfg="experiment.yaml", max_retries=1)
    print(batch.output_dir)
    for result in batch.run():
        print(result.run_id, result.status.value, result.output)
        for attempt in result.attempts:
            print(attempt.attempt, attempt.status.value, attempt.error_message)
```

- 首次执行初始数据为 `None`。由 Batch 执行时，每个阶段自动保存返回值与显式 checkpoint 状态（无 checkpoint 时为空状态）；重试跳过已完成阶段，恢复其输出和状态。
- 普通阶段失败会记录错误，在该顶层阶段的预算内重新入队。默认额外重试一次。`max_retries=0` 关闭普通失败的重试，取消和资源限额触发的重新调度单独处理。
- 重试保留 `run_id`。`ctx.attempt` 是当前顶层阶段的尝试编号，从 1 开始。`ctx.cfg` 深层只读，跨尝试的业务状态由 `ctx.checkpoint()` 显式保存。
- 中断主进程会结束调度并回收 worker。业务代码在 worker 内抛出 `KeyboardInterrupt` 或 `SystemExit` 时，该阶段记作取消并重新入队。
- 新 Stage 按参数异质性和资源装箱选择启动顺序；返回结果按配置顺序排列，每个结果保留全部尝试的状态、耗时和错误摘要；`output` 是最终成功尝试的返回值。事件可通过 `batch.recorder` 获取，支持传入自定义 Recorder。
- 成功结果保留最终快照的位置，每次访问 `result.output` 都从磁盘读取。失败或取消的尝试返回 `None`。主进程仍保留尝试摘要、阶段历史和记录器接收的事件。
- Batch 对象执行一次；继续已有实验使用 `Batch.resume()`，重新开始使用新的 Batch。也可以传入一份具体配置字典，创建只有一个实验的集合。

每次实际计算使用独立 worker。尝试摘要保存错误类型和消息。对文件、数据库或外部服务的写入可能在重试时再次发生，由业务代码处理。

## 命令行

终端运行与恢复共用 `module:attr` 定位流程。将一个 `Pipeline` 实例赋给模块属性即可。

```python
# mypkg/pipelines.py
import expman
from expman import Pipeline
from mypkg.stages import LoadData, Train, Eval

# 进程级默认值在这里配置，命令行没有对应旗标
expman.set(retries=3, coverage=0.9)

train = Pipeline([LoadData, Train, Eval])
```

```bash
# 创建并运行；pipeline 定位会记入 manifest；
# 默认输出目录 runs/<pipeline名>_<config名>，重名时追加 _2、_3……
expman run mypkg.pipelines:train experiments.yaml

# 独立启动运行看板，扫描 runs/ 下全部批次（运行中显示实时指标，已停止显示为已停止）
expman web

# 查看持久化进度与剩余时间区间（按 ID 或路径；只读，不影响运行中的 Batch）
expman status 0

# 不带参数：持续刷新 runs/ 下所有 Batch 的总览，直到 Ctrl+C
expman status

# 优雅停止运行中的 Batch：向运行进程发送 SIGTERM，
# worker 被终止、尝试记录为 cancelled、状态落盘后退出（退出码 143）
expman stop 0

# 从停止或中断处恢复；从 manifest 重新导入 pipeline，
# 恢复时会校验阶段顺序、类名与可获取的模块源码摘要
expman resume 0
```

- `runs/` 下每个 Batch 在 `runs/.batches.sqlite3` 中登记一个从 0 开始的稳定整数 ID。首次发现的批次按目录修改时间排序登记，移除批目录后保留其编号。`status` / `stop` / `resume` 都接受 ID 或目录路径。
- 在 pipeline 模块里调用 `expman.set(retries=..., coverage=...)` 可设置新建 Batch 的默认值，初始值分别为 1 和 0.8。Python 构造参数可以覆盖默认值。恢复时沿用批次清单里保存的设置。
- 进度显示默认关闭，需要时加 `--progress`；`run` 可用 `-o` 覆盖默认输出目录。
- CLI 的 `run` 和 `resume` 持有批次执行锁时会打开 `stop.fifo`。同一系统用户运行的 CLI 或看板可通过它探测批次是否仍在运行。向管道写入停止请求后，Linux 内核向持有读端的批次进程发送 SIGTERM。Python 调用可在主线程使用 `batch.run(handle_sigterm=True)` 启用相同机制。
- `run` 和 `resume` 会打印每个 run 的状态。批次全部成功时退出码为 0，有 run 最终失败时为 1，收到停止请求并中断时为 143。
- `resume` 要求 manifest 记录过 pipeline 定位（由 CLI 创建的 Batch 都会记录）；库 API 创建的目录请继续用 `Batch.resume(pipeline, ...)`。
- 看板是独立命令 `expman web`，扫描 `runs/` 下全部批次并常驻服务，与实验进程互不依赖；可选 `--port`（默认 `8765`，`0` 由系统分配、被占时最多尝试连续 10 个端口）、`--host`（默认仅回环）、`--root`、`--read-only`；详见[运行看板](#运行看板)。

## 运行看板

看板是一个**独立进程**，与实验进程互不依赖：它不训练模型、不参与调度，启动后扫描 `runs/` 下的**全部**批次，一次呈现所有历史与在飞实验。实验进程可以不存在，看板仍然可用；看板启动前后，实验也可以随时起停。

```bash
# 在项目根目录启动看板，默认 127.0.0.1:8765（从指定端口起最多尝试 10 个端口）
expman web

# 指定端口与绑定地址（0 由系统分配；默认仅回环）
expman web --port 8899 --host 127.0.0.1

# 指向别处的 runs/，或只读运行（禁用停止按钮）
expman web --root /path/to/runs --read-only
```

启动时终端打印看板地址。`/` 列出批次 ID、名称、状态、各类 run 的数量、已用与剩余时间。`/b/<ID>/` 展示该批次的阶段、worker、调度和资源详情。页面会注明数据来源。

| 情形 | 判定 | 显示 |
| --- | --- | --- |
| 运行中 | 进程存活且心跳新鲜 | 实时指标：调度、进度、逐卡「分配 vs 实际」、事件流 |
| 最后读数 | 有快照，停止管道未响应或心跳已过期，批次清单也没有更新 | 展示最近一次成功发布的读数，并标记数据来源 |
| 归档 | 没有可用快照，或最后一次采样后批次清单已有更新 | 从 `batch.pkl` / `timing.pkl` 重建的历史状态，实时面板置空 |

看板展示四类信息：

- **运行状态**：各 run 的运行/待定/成功/失败/取消计数，以及 Batch 累计运行时间与剩余时间区间。
- **阶段视图**：每个顶层 Stage 的剩余组数与单组耗时区间，两者相乘即该 Stage 的串行剩余估计；同时给出该 Stage 已测/复用/删失的样本数。
- **调度任务**。页面显示待定与运行计数、本轮主机和显存是否允许派发、实际派发数，以及启动、减载、完成和复用事件。逐阶段的队列分布和完整装箱计划保存在 `live.json`。
- **内存与显存**。主机与每张卡分别展示调度预算和采样占用，采样年龄 `age_seconds` 超过 2.5 秒时资源条变灰。每个 worker 的预算条另行标出可显示的主机或 MPS 限额。

实验进程默认约每秒把观测快照原子写入批次目录的 `live.json`，退出时再尝试发布一次。`EXPMAN_LIVE=off` 可关闭发布。发布线程在锁内读取调度状态和阶段进度，文件写入在锁外完成。看板优先读取这份快照。停止管道可连接且快照时间距现在不超过 3 秒时，来源标记为实时。快照缺失、不可用或最后读数早于最新批次清单时，看板从 `batch.pkl` 和 `timing.pkl` 重建归档视图。

看板默认仅绑定回环地址，允许通过 `POST /api/stop/<id>` 请求停止。使用 `--read-only` 可禁用该操作。发现新批次时，服务会在根目录登记编号。运行记录使用 pickle，只应向看板提供可信目录。页面静态资源随包提供。

### 结果分析

详情页顶部有「运行状态 / 结果分析」两个页签。「结果分析」对**当前批次**做简单的自变量—因变量分析，不需要批次跑完：

- **数据来源**：`batch.pkl`（Run 清单、状态、累计耗时）＋ 每个 Run 的 `config.pkl`（自变量）＋ `metrics.sqlite3`（`ctx.log_metrics` 记录的标量指标）。读不到配置的 Run 记为「跳过」并给出原因，不中断整体分析。
- **局部结果**：失败与未跑完的 Run 不参与统计但计入覆盖度（页面显示「Run N / M」）；每个指标各自计 `n`，缺该指标的 Run 按 pairwise 跳过并计入「缺失」。
- **逐对分析**：数值型配置项（≥3 个不同取值）报告 Spearman 秩相关 ρ，其绝对值即影响力度量；其余配置项按取值分组，报告每组均值/样本数、均值最高的取值，以及组间变异占指标总变异的比例（η²）。
- **跳过规则**：所有 Run 取值相同的常量项在页首列出一次、不参与比较；有效样本不足（<3）的配置项只计数不展示；每个取值只有一个样本时只列各组均值、不算影响力。
- **结果排序**：指标卡片按最大影响力排序，卡片内配置项也按影响力排序，直接回答「哪个配置项最影响这个指标」。

耗时以 `run_duration_seconds` 参与分析，汇总多次尝试。与它重名的业务指标，以及多个阶段记录的同名指标，用阶段路径区分。含点的字面键用引号区分，例如 `"train.loss"` 与嵌套的 `train.loss`。分析由 `GET /api/analysis/<id>` 提供，结果按批次输入文件的指纹缓存。点「重新分析」会重新请求，指纹变化时服务重新计算。这些都是对观测数据的描述（相关与分组差异），不是因果结论。

## 工作进程环境

调度器为每个新 worker 准备环境变量，再启动独立解释器。

- **线程池**。默认将 OpenMP、MKL、OpenBLAS 等受支持线程池的环境变量设为 4。实际线程数由各库决定。
- **HF 镜像**：`HF_ENDPOINT` 未设置时指向 `https://hf-mirror.com`，模型下载不依赖到 huggingface.co 的直连。

shell 已导出的设置优先。设置任一受支持的线程池变量（如 `OMP_NUM_THREADS` 或 `VECLIB_MAXIMUM_THREADS`），框架便保留整组线程设置。已有 `HF_ENDPOINT` 也会保留。以下变量可以调整默认行为。

```bash
EXPMAN_THREADS=8 expman run mypkg.pipelines:train experiments.yaml   # 改上限
EXPMAN_THREADS=off expman run ...                                    # 关闭线程上限
EXPMAN_HF_MIRROR=off expman run ...                                  # 不动 HF_ENDPOINT
EXPMAN_HF_MIRROR=cn expman run ...                                   # 用默认镜像
EXPMAN_HF_MIRROR=https://mirror.internal expman run ...              # 指定端点
```

本次尝试的 `bootstrap.pkl` 中，`environment` 字段记录 expman 添加的线程池和镜像设置。从 shell 继承的设置可在 worker 的进程环境中查看。`EXPMAN_THREADS` 取值非法时，新建 Batch 会在创建输出目录前报错。

## 阶段快照与 checkpoint

Stage ID 是 Pipeline 中从 0 开始的位置，通过 `ctx.stage_id` 读取。返回值和状态量一起原子保存，保存成功才标记阶段完成；不可序列化的返回值或状态量会使该阶段失败并进入重试流程。

### 显式 checkpoint

在 `process()` 内决定循环、状态形状和存档时机。用 `ctx.resume_range(start, stop)` 获得可恢复的迭代器，用 `ctx.checkpoint(position, state)` 保存下一次要执行的位置和可序列化的状态。

```python
class Train(Stage):
    def process(self, data, ctx):
        state = ctx.checkpoint_state
        if state is None:
            state = {"epoch": 0, "model": build_model(ctx.cfg)}

        for epoch in ctx.resume_range(0, ctx.cfg["epochs"]):
            train_one_epoch(state["model"], data)
            state["epoch"] = epoch + 1
            ctx.checkpoint(epoch + 1, state)
            if should_stop(state["model"]):
                break

        return state["model"]
```

`resume_range(start, stop)` 在首次执行时从 `start` 开始；若存在可读 checkpoint，则从 `max(start, checkpoint 记录的位置)` 开始，保证已完成的位置被静默跳过。`checkpoint(position, state)` 把位置和状态原子写入该阶段的 `checkpoint.pkl`。下一次尝试通过 `ctx.checkpoint_state` 取回保存的值。随机状态在阶段入口自动重置到最近一次 checkpoint 的存档时刻，因此失败迭代中的随机消耗不会被继承。

每个阶段只保留最近一次成功写入的 checkpoint。完成快照包含最后保存的业务状态和阶段返回值。直接运行 `Pipeline.run()` 时，`resume_range` 从起点迭代，`checkpoint_state` 为 `None`。在这种上下文里调用 `checkpoint()` 或 `log_metrics()` 会抛出 `RuntimeError`。

对于 dataloader、文件句柄等进程绑定资源，应在 worker 中重新创建，并在 checkpoint 里保存重建所需的数据。

Python 创建 Batch 时默认目录为 `runs/<随机 UUID>/`，可通过 `output_dir=` 指定一个尚不存在的目录。配置在 Experiment 创建时保存一次，运行中的映射和列表使用只读包装；读取、索引、遍历照常，可变数据（张量、计数器等）定义为状态量。

```python
# 原进程中创建并运行
batch = Batch(pipeline, "experiment.yaml", output_dir="runs/my-batch")
batch.run()

# Ctrl+C 或进程退出后，在新进程中提供相同的流程定义
batch = Batch.resume(pipeline, "runs/my-batch")
results = batch.run()
```

恢复使用持久化配置、实验 ID、尝试历史和待执行队列，不依赖原 YAML。中断不消耗失败重试机会；突然退出的未结束尝试记录为 cancelled，无法获知的耗时记为 0。恢复检查阶段类、顺序及可获取的模块源码摘要。外部依赖和动态生成的代码由用户管理。同一目录只允许一个进程执行。

默认保存器 `PickleSerializer` 支持可 pickle 的 Python 对象，**只加载可信的本地快照**，并保持兼容的 Python、依赖及设备环境。文件句柄、生成器等不保证可保存。可实现 `Serializer.dump/load` 并在创建和恢复 Batch 时传入同一种 `serializer`。阶段、checkpoint 和 Batch 清单保存失败会报错；可选 Recorder 的普通故障会记录日志，默认指标事件只存在于当前进程内存中。

## 持久化数值指标

在由 `Batch` 管理的阶段内，使用 `ctx.log_metrics()` 保存嵌套指标：

```python
ctx.log_metrics(
    {"train": {"mse": 0.1, "mae": 0.2}, "val": {"mse": 0.15}},
    step=epoch,
)
ctx.log_metrics({"test": {"mse": 0.08}})  # 汇总指标
```

Batch 内所有实验共用输出目录中的 `metrics.sqlite3`。数据库在首次非空写入时创建。每次调用先校验整棵字典，再以一个事务写入，返回即已提交。写入失败会使阶段失败，并进入 Batch 的重试流程。

指标名为非空字符串，叶子为有限实数（不接受布尔值、NaN 和无穷值），统一保存为 SQLite REAL（双精度浮点数）。`step` 为非负的 64 位有符号整数或省略。空字典不产生记录。嵌套深度受 Python 递归限制。

唯一键为实验 ID、完整阶段位置路径、step 和完整指标路径。同键写入覆盖数值、尝试编号和 UTC 更新时间；本次未传入的指标保留。恢复时保留所有旧记录，包括 checkpoint 之后的指标，重新运行到相同位置时再覆盖。恢复本 run 的已完成快照时沿用已有指标。跨 run 复用共享前缀时，框架把节点中的指标写入目标 run。

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

## 进度与剩余时间

```python
batch = Batch(pipeline, cfg="experiment.yaml", estimate_coverage=0.8)
print(batch.estimate())  # 有待执行阶段，尚无耗时样本
results = batch.run(refresh_interval=1.0)
print(batch.estimate())  # 全部结束时为 00:00:00～00:00:00
```

Python 的 `Batch.run()` 默认向标准错误输出进度，CLI 的 `run`、`resume` 通过 `--progress` 开启。每个阶段显示已完成组数与总组数，每张卡显示活跃的 worker 数。下面的示例中，两个配置共享阶段 0，阶段 1 还剩一组。

```text
运行中 | Stage0:1/1 Stage1:1/2 | GPU0:1运行 | 已运行 00:00:35 | 预计剩余 00:00:02～00:00:04
```

耗时采用 `DD:HH:MM`。例如 `00:00:35` 表示 35 分钟。批次累计运行时间跨 resume 延续，两次运行之间的停机时间排除在外。批次约每 5 秒保存累计时间与当前可用的 ETA，正常收尾时也会保存。进程意外退出时，以最近一次成功写入的记录为准。

`batch.estimate(coverage=0.95)` 计算未完成阶段组的完工时间区间。覆盖参数须大于 0 且小于 1，用于校准单阶段耗时区间。单次查询后，进度显示和看板仍使用批次配置的覆盖率。模型用配置距离加权的历史对数耗时估计中心，用留一残差估计区间。取消的样本提供耗时下界。并行估计考虑阶段依赖和可用槽位，分别用各阶段的耗时下界、上界进行贪心排程。整批区间取两次排程完工时间中的较小值与较大值。缺少必要的耗时数据时显示问号，全部结束时显示零。[调度与估时的源码导读](DESIGN.md)

## GPU 派发与资源限制

- `device` 是整批共用的物理 GPU 编号列表。调度器通过设备 UUID 为每个 worker 设置一张可见卡，业务代码可以使用 `cuda:0`。worker 在运行用户代码之前建立 CUDA 主上下文并核验 MPS 注册。只做 CPU 计算的 Stage 也会建立该上下文。
- 调度器每轮采集主机可用内存与设备空闲显存，扣除主机预留量和已有 worker 尚未占用的预算，再选择可同时运行的任务。主机预留 1 GiB。没有相应峰值样本时，主机和显存的初始预算分别为 1 GiB。[资源默认值](src/expman/resources/devices.py)
- worker 的资源上限在估计峰值上留有余量。显存上限通过 MPS 请求，主机内存上限取决于可用的 cgroup 机制。缺少独立 cgroup 读数时，worker 尝试读取自身的 RSS 和历史峰值。看板区分预算、实际采样和已核验的上限。
- 调度器根据采样结果决定是否暂停派发或减载。被确认为资源限额导致的失败会重新估价并入队。每次尝试的标准输出和错误输出位于 `experiments/<run_id>/attempts/stage-<index>/<attempt>/output.log`。
- worker 需要可导入的顶层 Stage 类。入口脚本使用 `if __name__ == "__main__":` 保护启动代码，Pipeline 和自定义 serializer 需能通过 pickle 传入子进程。worker 的标准输入用于检测父进程退出。

## 随机种子与随机状态恢复

每个实验启动时读取必填的根部 `seed`，并在构造和执行 Stage 前设置 Python `random`、NumPy 全局随机生成器，以及 PyTorch CPU/CUDA 随机种子。种子必须是 `0` 到 `2**32 - 1` 的整数，不接受布尔值。嵌套配置中的同名字段由用户代码解释。

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

每个 run 将初始化后的随机状态原子写入 `experiments/<run_id>/rng_initial.pkl`。重试和恢复读取此记录，不重复设种子；随后按已有阶段快照和 checkpoint 恢复到相应位置。阶段完成快照及 checkpoint 都包含独立的 `rng_state` 字段，与业务状态量在同一文件、同一事务中保存。框架只保留最新一份 checkpoint，阶段完成时把它删掉：完成快照已经是恢复的依据，留着中间 epoch 的模型只是在占盘。

复用本 run 已完成阶段时恢复该阶段结束时的随机状态；失败阶段有 checkpoint 时，框架把随机状态重置到存档时刻，没有 checkpoint 则从阶段入口的随机状态重跑。恢复尝试从 `ctx.resume_range()` 返回的位置继续，checkpoint 之前的随机数消耗自然不计入续跑序列；最近一次 checkpoint 之后的迭代会重新执行。

随机状态记录覆盖 Python 全局生成器（含 Gaussian 缓存）、NumPy 全局 RandomState、PyTorch CPU 及全部可见 CUDA 设备的生成器。NumPy 数组转为普通列表、PyTorch ByteTensor 转为 bytes 存储，因此仅查看快照元数据不会为了随机状态反序列化 CUDA Tensor。保存可见 CUDA 状态会初始化相关生成器，需要可用的 CUDA 环境；GPU 调度器已在子进程启动前限制设备可见范围。

无法读取 checkpoint 或其记录结构无效时，框架发出 `RecoveryWarning` 并从该阶段开头执行。恢复随机状态失败会报错。恢复所需的随机库缺失、CUDA 状态数量与可见设备不一致等不兼容情况会报错。

独立的 `random.Random`、NumPy `Generator/default_rng`、`torch.Generator`、DataLoader 工作进程及第三方库的状态由用户保存。此功能不自动开启确定性 GPU 算法、不控制 Python hash 随机化，也不保证跨库版本或硬件逐位一致；参见 [PyTorch 可复现性说明](https://docs.pytorch.org/docs/stable/notes/randomness.html)。直接使用未受 Batch 管理的 Pipeline/Stage，不会自动设置全局种子。

## 同 Batch 连续前缀复用

同一 Batch 内可以共享已完成的连续阶段前缀。根部 seed 和 Pipeline 签名隔离缓存。每个 Stage 可用 `config_dependencies(cfg)` 声明影响结果的配置路径。共享命中要求上游节点相同，且当前阶段声明的依赖路径和对应值与已保存记录完全匹配。

例如两个实验仅 model 不同，阶段 0 声明只依赖 data，阶段 1 声明只依赖 model，那么第二个实验可以复用阶段 0。命中时恢复返回值和随机数状态，并将该阶段的数值指标写入当前 run 的 SQLite 记录。

依赖声明与配置树同构：节点为 `True` 表示依赖该子树，为映射则递归到子键（序列用整数下标），缺省或 `False` 表示不依赖，返回 `True` 表示依赖完整配置。声明方法接收只读的 `cfg`，因此可以按取值选择分支——例如 `Baseline` 为 `B1` 时只声明 `imputer`，为 `B2` 时同时声明 `predictor`。默认返回 `True`：读取 `ctx.cfg` 却未重写该方法的 Stage 会保守地依赖完整配置，不会错误复用。配置中的字段必须显式声明，`get()` 和 `in` 判断抛出 TypeError，索引不存在的字段抛出 KeyError。共享缓存查找和 checkpoint 恢复会检查已记录依赖的值。本 run 的已完成阶段按流程签名和阶段位置恢复。

共享快照位于 Batch 的 cache 目录，各 run 的 completed.pkl 保存引用；移动实验记录时应保留完整 Batch 目录。自身已有快照和 checkpoint 优先恢复。调度器在同一就绪组里先派发一个代表，其他成员在它发布完成快照后复用结果。嵌套 Pipeline 的依赖由外层 Stage 的声明覆盖。

复用以相同上游、相关配置和随机状态产生一致结果为前提；文件内容等外部输入发生变化时，应将能区分输入的标识写入阶段依赖的配置中。跳过阶段不会重新执行其中的外部副作用。共享范围限于同一个 Batch。

## 可恢复的阶段诊断计时

每个阶段的完成快照及 checkpoint 都包含 `elapsed_seconds`（秒）。构造 Stage 之前启动单调时钟；保存时将恢复的累计值与本次执行时长相加，与进度写入同一原子记录。计时字段由框架管理，不占用状态量。

例如 checkpoint 记录 100 秒，随后运行 20 秒后中断；恢复后再运行 30 秒完成，阶段快照记录 130 秒。没有 checkpoint 的失败阶段重跑时从零计时。停机时间和恢复检查点的加载时间不计入累计值。保存时取写入前的时间截点，因此当前保存操作的耗时不在该记录内；不中断继续运行时，它会进入下一次记录的执行时长。嵌套阶段独立计时，外层耗时包含内层执行，不能直接将各层耗时相加。

共享复用保留源快照的 `elapsed_seconds`。worker 加载已有快照时会在本 run 的 `status.pkl` 记录本次恢复用时 `restore_seconds`。主进程直接根据元数据复用时，这一字段记为 0。累计耗时必须是有限的非负数。checkpoint 中该字段无效会告警并从阶段开头执行，已提交快照中该字段无效则抛出 `StorageError`。

Batch 记录各次尝试占用 worker 的时长，包括有记录的失败和中断。阶段快照的累计耗时用于查看已保存进度所花的时间。


## 开发与验证

默认测试和提交检查都会运行两项真实 CUDA/MPS 测试，自动选择可见且显存足够的 GPU 中利用率最低的一张。单独运行这两项测试可执行

```bash
uv run --no-sync pytest tests/gpu/
```

测试通过 CLI 启动 worker，验证 CUDA 计算结果、跨配置复用、故障重试，以及写入 checkpoint 后主动停止并恢复。缺少可用 GPU 或 MPS 会使测试失败。[选卡条件与提交检查](CONTRIBUTING.md#真实-gpu-验证)
