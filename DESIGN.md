# expman — 实验管理库设计文档

版本：v0.7（草案） · 日期：2026-09-06 · 状态：设计中，待逐节讨论

> 版本记录：
> - v0.7（2026-09-06）：管道化实验类（§4.1：cfg_cls + pipeline + Stage 契约 + transform 缓存）；BatchSpec dims 改为 cfg 字段路径（§3）；整批 prep 唯一化（§5.3）；US-10、M14 ✅，M2/M8 ✅
> - v0.6（2026-09-06）：对接定为 BaseRun 基类双轨（§4.1）；时间单位统一为 step（§3、§5.2/5.3、§6、§7、§8、US-2/5/7/8）；M13 ✅
> - v0.5（2026-09-06）：明确系统边界与职责（§4.1 重写）；workload 泛化 train/inference（§3、§5.3）；diversity-first + cost surrogate（§5.2）；US-8/9、M11/M12 ✅
> - v0.4（2026-09-06）：时间评估改为"滚动式测量窗口"（§5.2 重写、§5.3 简化、§3 probe.mode、US-7、M10 ✅）
> - v0.3（2026-09-06）：合入自适应并行调度器（US-6、§4.4.1、§6 增量、§11 M9 ✅）
> - v0.2（2026-09-06）：附录访谈产出（§10 US-1..5、§11 M1-M8）
> - v0.1（2026-09-06）：初稿（§1-§9）

> 一份"批量机器学习实验管理库"的设计。差异化核心：**用分钟级试探实验（Probe）外推整批实验的总耗时与可行性**。开放问题见 §9（OQ-1..10），所有结论均可推翻重议。

---

## 1. 背景与目标

### 1.1 要解决的问题

研究者日常批量实验的三个痛点：

1. **开工前可行性未知**：无法回答"这 50 组实验跑完要多久、值不值得跑"，只能拍脑袋。跑了 3 天才发现总共要 10 天，沉没成本已发生。
2. **批量启动靠手写**：for 循环 + nohup/tmux + 手工改配置；失败重试、断点恢复、并发控制全靠人肉。
3. **结果散落**：每个实验的 config / metric / artifact 散在各目录，对比时手工抄表。

### 1.2 目标

- **时间评估（核心）**：每 run 前 K 个 step 为内嵌测量窗口（warm start，几乎零开销），滚动给出整批实验的期望总耗时与区间（P10/P50/P90），对照预算/期限输出**可行性判定**；不可行时给出量化的削减建议，且止损成本最低。
- **批量执行抽象**：任务清单 + 状态机 + 可插拔后端（v0.1 单机多进程；Slurm/云留接口）。
- **结果自动归档**：config 快照、逐 epoch metric、耗时、artifact 路径统一入库，一键出对比表。
- **对训练代码零框架侵入**：模型层与训练层全归用户代码，库不提供 Trainer/训练循环；接入面 = BaseRun 基类（推荐，约定内建）或 4 条程序边界约定（纯黑盒兼容，§4.1），一次接入、所有项目通用。

### 1.3 非目标（v0.1 明确不做）

| 不做 | 理由 |
|---|---|
| HPO 搜索算法 | Optuna 的活；本库执行"给定的一批实验"并预估，搜索策略在上层 |
| 训练框架（模型/优化器） | 那是用户代码 |
| 分布式训练运行时 | v0.1 后端 = 多进程独占设备 |
| Web UI / 平台 | CLI + markdown 报告足够（v0.1） |

### 1.4 术语表

| 词 | 定义 |
|---|---|
| Run | 一次独立训练执行 = 一份渲染完成的 config + 状态 |
| Batch | 一批要执行的 Run = BatchSpec 实例化结果 |
| BatchSpec | 批量实验的声明式定义（config 模板 + 变化维度 + 预算） |
| Probe | 试探性运行：run 前 K step 的测量窗口（warm start），产出 s/step / 显存峰值 / 早期曲线；every_run 每 run 都有，per_family 每族代表有 |
| Stage | 管道阶段：transform（数据变换，无状态、只传数据、产物可缓存）或 step_loop（Train/Inference，由基类骨架驱动 step 循环） |
| workload 族 | 单位时间成本相同的 Run 群体（通常 = 同模型架构 × 同数据规模；族内差异仅超参/seed） |
| Estimate | 时间外推输出：总卡时/墙钟 + P10/P50/P90 + 方法说明 |
| 卡·小时 | 资源口径：1 卡跑 1 小时。墙钟 ≈ 卡时 ÷ 并行设备数（理想线性） |

---

## 2. 系统架构

```
BatchSpec(yaml) ──► Planner ──► 任务清单(SQLite/JSONL)
                       │  ▲            │
                 probe/估时      Executor（后端抽象）
                       ▼  │            ▼
                  Estimator ◄──实际耗时回填(校准)──► 训练进程 × N（零侵入约定）
                                                      │
                                                 Recorder → SQLite + artifacts/
                                                      │
                                            report：对比表 + 预估vs实际
```

分层单向依赖：**配置层**（BatchSpec schema）→ **计划层**（Planner / Estimator / 可行性判定，本库灵魂）→ **执行层**（Executor 后端）→ **记录层**（Recorder / Reporter）。每层可单独替换。

---

## 3. BatchSpec：批量实验的声明式定义

```yaml
name: ts_impute_sweep

experiment:
  module: experiments.ts_impute     # 实验类（cfg_cls + pipeline()，§4.1 轨道 1，推荐）
  class: TSImputeExperiment
  # 轨道 2（纯黑盒，无实验类）时改用：
  # config:
  #   base: configs/ts_impute.yaml  # 用户自己的配置文件
  #   entry: train.py               # 训练入口

# 变化维度（cfg 字段覆盖路径）：笛卡尔积 → Run 集合
dims:
  - cfg_path: model.name
    values: [SAITS, BRITS, TimesNet]
  - cfg_path: data.missing_rate
    values: [0.1, 0.3, 0.5]
  - cfg_path: seed
    range: [0, 2]                  # 闭区间整数

# 每 run 的 workload 规模（总时长 = total_steps × s/step，见 §5）
workload:
  type: train                      # train | inference
  epochs: 200                      # type=train：总 epoch 数（基类按 steps_per_epoch 折算）
  # steps: 40000                   # 或直接声明总 step 数（与 epochs 二选一）
  patience: 30                     # type=train：早停耐心（epoch 计，基类折算为 step）
  batches: 5000                    # type=inference：总 batch 数 = step 总数（无 epoch 概念）

# 设备需求（Executor 据此调度）
device:
  per_run: 1 gpu                   # 单 run 资源需求
  concurrent: auto                 # auto=自适应并行（§4.4.1）；或固定并发数
  margin: 0.10                     # 显存安全余量（防相位同步抖动）

# 试探实验与可行性判定
probe:
  enabled: true
  mode: every_run                  # every_run=每 run 都测（默认，精确）
                                   # per_family=族采样代表测（run 数百+时，见 §5.2）
  steps: 100                       # 测量窗口 = 前 K 个 step（且测量时长 ≥3s，§5.2）
  max_minutes: 10                  # 单 run 测量窗口硬上限（护栏）
  decision_after: 6                # 前 W 个多样化 run 测完即输出首个整批预估（决策点，§5.2）
feasibility:
  budget: 72h                      # 墙钟期限；也可写 card_hours: 500
```

渲染规则：dims 笛卡尔积 → run 清单；每个 run 的最终 config = base ⊕ 行内 override，**config 快照强制入库**（可复现）。

---

## 4. 模块设计

### 4.1 系统边界与职责（v0.5）

**边界一句话：模型层与训练层全归用户代码；expman 是训练程序黑盒之上的编排/观测/预估层**（≈ Slurm 之于作业、CI 之于测试命令）。库**不提供 Trainer、不实现训练循环**——研究代码的训练逻辑千差万别，接管训练循环意味着每个新项目都要重构进库的框架，违背北极星"接入一次、少烦心"。

| 层 | 谁负责 | 内容 |
|---|---|---|
| 模型层 + 训练层 | **用户代码** | 模型架构、loss、训练循环、早停、验证、ckpt 读写、推理逻辑 |
| 编排/观测/预估层 | **expman** | 批量规划、调度/并行/重试、时间与显存观测、预估与可行性、断点编排、结果库、报告 |
| 接入面 | 双轨（下） | 轨道 1：BaseRun 基类，约定内建、规范代码；轨道 2：纯黑盒 4 条约定——均在程序边界，无框架继承 |

**接入双轨**：

- **轨道 1（推荐）｜`BaseRun` 基类：约定内建，规范化代码撰写**。子类继承并实现最少钩子；main 入口、step 循环、上报、测量窗口、ckpt、resume、早停全部由基类完成——约定写在基类里，不可能被违反；AI 生成代码也只需"继承 + 实现 run_unit"，结构天然统一（M8）。

  ```python
  class BaseRun:
      def build(self, cfg: dict) -> None: ...      # 可选：模型/优化器/数据；可声明 self.steps_per_epoch
      def run_unit(self, step: int) -> dict: ...   # 必需：完成第 step 步（train: 1 次前向+反向；
                                                   #        inference: 处理 1 batch），返回 metrics
      def save_extra(self) -> dict: ...            # 可选：自定义 ckpt 内容
      def load_extra(self, extra: dict): ...       # 可选：恢复自定义状态
  ```

  默认骨架：基类实现 run_loop（step 循环 + emit JSON + ckpt 落盘 + step 级早停 + resume 恢复）。**逃生舱**：非常规流程（多阶段/GAN/自定义调度）可 override run_loop，基类退化为工具函数（emit_step/save_ckpt/should_early_stop），约定仍在。epoch 语义降级为子类内部概念：run_unit 内用 `step % self.steps_per_epoch` 判 epoch 边界（验证/scheduler）。库附**示例子类**作模板（原"参考模板"演进）。

**v0.7 管道化形态（推荐写法，v0.1 起实现）**：实验 = **实验类**（配置集中一处 + 管道定义），BaseRun 为执行器：

```python
@dataclass
class TSImputeConfig:             # 全部实验配置集中（数据/预处理/模型/训练分区）
    data:  DataConfig  = DataConfig(path="physio2012", missing_rate=0.3)
    model: ModelConfig = ModelConfig(name="SAITS", d_model=64, n_layers=2)
    train: TrainConfig = TrainConfig(lr=1e-3, patience=30)

class TSImputeExperiment(BaseRun):
    cfg_cls = TSImputeConfig
    def pipeline(self) -> list[Stage]:   # 管道：模块之间只传数据
        return [LoadRaw(), InjectMissing(), Normalize(), Window(),
                Train(), Evaluate()]
```

Stage 契约：`kind: transform | step_loop`；`run(data, ctx) -> data`。
- **transform 阶段**（LoadRaw/InjectMissing/Normalize/Window）：无状态数据变换；产物按 `(cfg_hash, 阶段序号, stage.version)` 落盘缓存——批量内同配置只真跑一次，其余命中缓存（27 run × 3 missing_rate → 预处理只付 3 次）
- **step_loop 阶段**（Train/Inference）：由基类骨架驱动 step 循环，测量窗口/ckpt/resume/早停机制全部不变，run_unit 语义下沉为该阶段内部实现
- 多阶段训练 = 管道串联两个 Train 阶段（原"逃生舱"场景多数自然消解）
- 模型等有状态对象走 ctx，不混入数据流；"模块间只传数据"语义干净
- 直接实现 run_unit 的旧写法仍兼容（管道是默认推荐，不是强制）

- **轨道 2（兼容）｜纯黑盒**：老代码/一次性脚本不继承也能跑，手动满足 4 条约定：
  1. **入口**：`train.py --config <file.yaml>`（退出码 0/非 0）或 `train(cfg: dict)`；
  2. **step 事件**：每完成 N step（默认 10，可配）向 stdout 打一行 JSON：`{"step": 137, "loss": 0.12, ...}`——**不可省**（外部无法自行得知 step 边界）；`final_metrics.json` 可选收尾快照；
  3. **checkpoint 落盘约定路径**：expman 发现 ckpt 即传 resume 标志（step 级续跑，US-2）；
  4. **total_steps 声明**：workload 规模在 BatchSpec 声明（§3），expman 不解析用户 config 内部结构。

### 4.2 Planner

BatchSpec → 任务清单 + probe 建议集。职责：维度展开、run id 分配（`<batch>/<dims 摘要>/<seed>`）、清单持久化。v0.1 无 Run 间依赖（无 DAG；级联/依赖实验 → OQ）。

### 4.3 Estimator（时间评估）→ 见 §5，本库灵魂。

### 4.4 Executor（后端抽象）

```python
class Backend(Protocol):
    def submit(self, run: Run) -> Handle: ...
    def wait(self, handles, timeout) -> list[RunResult]: ...
    def kill(self, handle) -> None: ...
```

- **v0.1 `LocalProcBackend`**：N 进程按设备数并发；失败自动重试（次数可配）；stdout 实时入库。
- **状态机**：`pending → probing → running → done | oom_retry | failed | cancelled`（probing=峰值未定；oom_retry=放回池尾），状态落 SQLite，进程重启可 `--resume`。
- 预留（只留接口 + 设计说明，不实现）：`SlurmBackend`（sbatch 包装）、`CloudBackend`。

#### 4.4.1 自适应并行调度器（US-6，v0.3 合入）

价值：AutoDL 按卡计费，小实验（6-8G）独享 24G 卡浪费；同卡多进程并行省卡时，但会争算力。

- **显存观测（零侵入）**：外部采样器每 2s 查 per-process 显存（nvidia-smi / pynvml），训练代码零改动；每 run 维护观测峰值 peak_i；probe 的族峰值显存（§5.2 已记录）成为调度先验。
- **峰值稳定判定**：连续 E=3 个 epoch 未创新高即定（封顶 5 epoch 或总 epoch 20%）；验证在 epoch 末的注意按 epoch 边界采样。
- **启动决策（贪心装箱）**：Σ(同卡稳定峰值) + 新 run 需求 × (1+margin) ≤ 卡容量 C（C 自动探测）。需求已知（族有 probe/历史）直接判；未知则独占试探启动。
- **OOM 分类**：并发挤爆（崩溃时并发>1）→ 放回池尾 + 记录失败组合 + 该族并发上限降级（防活锁）；独占仍 OOM（放回重试一次仍死）→ `failed(配置不可行)`，报告"需 ≥X GB 或换卡"。
- **算力竞争**：同卡并行拖慢系数 α 由校准表学习（v0.1 默认 0.8）；ETA 按有效并发 = 并发数 × α 折算。
- `status` 显示：每卡占用、并发数、排队数、动态 ETA。

### 4.5 Recorder & Reporter

- **Recorder**：SQLite（runs / metrics 时序 / final_metrics / estimates / calibration）+ `artifacts/` 文件树（config 快照、best 权重路径登记——只登记路径，不拷权重）。
- **Reporter**：`expman report` 输出 markdown/CSV 对比表：各 run 最终 metric、实际耗时、单位时间、**预估 vs 实际 ratio**。

---

## 5. 时间评估模块（核心）

### 5.1 目标

对 Batch 输出（**随进度滚动收敛，非一次性**）：

1. 总卡时估计的 **P10 / P50 / P90**：首个决策点（前 `decision_after` 个 run 测完）即出初版，之后逐 run 实测替换预填，精度单调改善；
2. 按可用设备数 N 的墙钟换算（含 §4.4.1 并发拖慢修正）；
3. **可行性三态判定**：✅ 可行 / ⚠️ 勉强（附建议）/ ❌ 不可行（附量化削减建议）——决策点输出，此时止损成本最低。

### 5.2 测量窗口、diversity-first 与 cost surrogate（时间评估执行机制）

**原则 1｜测量内嵌于正式运行（warm start），不设独立 probe 阶段。** 每个 run 的前 K 个 step（K 默认 100，且测量时长 ≥3s）是测量窗口，窗口结束不中断，直接继续正式运行。**step 是唯一时间单位**：train 1 step = 1 次前向+反向；inference 1 step = 1 个 batch。窗口产出：s/step 均值与 CV（跳过前若干 warmup steps；取后 1/4 段防数据缓存虚高）、显存峰值（→ §4.4.1 装箱先验，几分钟内即可用）、早期 loss 曲线（→ v0.2+ 收敛/早停预判）。

**原则 2｜diversity-first 启动序。** run 执行顺序由 Planner 规划而非书写顺序：高优先级 run 插队；其余按**空间填充**排序——让前 W 个被测量的 run 覆盖参数空间的不同区域（不同模型 × 不同规模数量级 × 不同 batch）。warm start 使重排序免费（不增总时长），换来决策点预估更快收敛：避免顺序跑时"前 3 个同类小模型 → 整批预估严重低估/高估"。

**原则 3｜预填用 cost surrogate（用部分参数组合的时间估计其余组合）。** 已测 run 用实测；未测 run 用代理模型预填：
- 特征：(模型类别, log 参数量, log 数据规模, batch_size, 序列长)；目标：log(s/step)——与 total_steps 解耦（total_steps 是 BatchSpec 声明值，不参与外推）
- v0.1 实现：log-log 线性/幂律回归；样本攒够后可换更强模型（原 OQ-6 激活）
- **安全边界**：(a) 只做预填与早期预估，实测永远优先替换；(b) 留一 CV 质量监控：status 显示 `预估置信度 ±X%（实测 W/N，CV-RMSE 0.25）`——样本少时明示"估得粗"，不假装精确
- `per_family` 模式（run 数百+ 不全测）下，未测 run 的继承值同样走 surrogate

**滚动收敛与决策点**：首个决策点 = 前 `decision_after`（默认 6）个多样化 run 完成窗口 → 输出整批 P10/P50/P90 与可行性三态——若不可行，止损只浪费约 N×K 个 unit。此后实测逐 run 替换预填，精度单调改善，"已确认占比"（§5.3）指示可信度。

**护栏**：单 run 测量窗口超 `max_minutes`（默认 10 min）即截断，用已测步数外推。

### 5.3 外推与聚合（v0.5）

统一公式：**T_run = T_prep（缓存未命中时）+ total_steps × s/step + T_eval**。total_steps 由 BatchSpec 声明（train: `epochs × steps_per_epoch` 折算或 `steps` 直写；inference: `batches` = step 总数），s/step 来自实测（测量窗口）或预填（cost surrogate，§5.2）。**批量内共享缓存唯一化**：同 cfg_hash 的 transform 产物只计一次 prep 时间（27 run × 3 missing_rate → prep 只付 3 次）；prep/eval 耗时在首次缓存未命中时测量入校准。整批预估 = Σ T_run，随实测替换预填滚动收敛。

- **train**：total_steps = epochs × steps_per_epoch（steps_per_epoch 由子类 build 中声明 `self.steps_per_epoch`；早停修正见下）
- **inference**：total_steps = batches，无早停（即全部）
- **早停修正（仅 train）**：总时长 ≠ Σ(total_steps×s/step)。patience 以 epoch 声明、基类折算为 step（patience × steps_per_epoch）后做 **step 级早停**；情景模型：P50 早停率 = 先验 `early_stop_frac`（默认 0.3，BatchSpec 可覆盖），早停 run ≈ 0.15 × 全量时长；P90 全跑满（悲观）；P10 早停率 × 1.5（乐观）
- **per_family 模式**（run 数百+）：未测 run 的 s/step 取族代表或 surrogate 值；跨族外推仅以 surrogate 预填形式存在（早期预估用），最终值永远来自实测
- **输出**：`Σ卡·小时 {P10, P50, P90}` + "已确认占比"（实测 run 卡时 ÷ 总预估，滚动收敛的置信指标）；墙钟 = 卡时 ÷ 有效并发（= 并发数 × α，见 §4.4.1）。

### 5.4 可行性判定（纯函数，可单测）

```
assess(estimate, budget) -> Verdict
✅ 可行    P90 墙钟 ≤ budget
⚠️ 勉强    P50 ≤ budget < P90   → 建议：加设备 N / 减 epochs / 提高早停阈值
❌ 不可行  P50 > budget          → 自动给出量化削减选项（如 model 维度砍半后重估，秒级）
```

### 5.5 校准闭环（长期差异化卖点）

Batch 跑完后：实际总卡时、各族实际 s/step 回填 `calibration` 表 → 下次同族 estimate 乘 EMA(ratio)。**库越用越准**；`report` 输出"预估 vs 实际"散点供审计，估计误差作为 Estimator 的公开质量指标。

### 5.6 与 AutoML 方法的边界（参考）

学习曲线外推 / 预算分配（ASHA、freeze-thaw）解决"训练**中**何时停"；本模块解决"开工**前**要多久"，二者正交。族间元模型已由 §5.2 cost surrogate 承担早期预填角色（v0.5，原 OQ-6 激活）；其幂律外推文献仍可借鉴。

---

## 6. 数据模型（SQLite v0.1）

```
batches        (id, name, spec_yaml, created_at)
runs           (id, batch_id, run_name, config_json, status,
                device, started_at, finished_at, wall_sec, exit_code,
                probe_of_batch_id NULL, workload_group,
                peak_mem_mb, oom_count, concurrent_siblings)
metrics        (run_id, step, key, value)          -- 逐 step 时序
final_metrics  (run_id, key, value)
estimates      (batch_id, kind[probe|final], per_group_json,
                card_hours_p10/p50/p90, method)
calibration    (batch_id, group, est_card_hours, actual_card_hours, ratio)
artifacts      (run_id, kind, path)                -- 只登记路径
oom_lessons    (workload_group, card_id, max_concurrent)   -- OOM 教训，防活锁
stage_cache    (cfg_hash, stage_idx, version, path, created_at)  -- transform 产物缓存索引
-- calibration 增列：concurrency（同卡并发数）、slowdown_alpha（算力拖慢系数，学得）
```

---

## 7. CLI 草案

```
expman plan   batch.yaml             # 渲染任务清单 + diversity-first 执行序
expman check  batch.yaml --budget 72h   # 可行性判定（纯函数，用测量/校准数据）
expman run    batch.yaml [--resume]  # 批量执行：BaseRun 驱动 + 测量窗口滚动预估
expman status [batch]                # 进度：step X/Y、并发、双 ETA、预估置信度
expman report [batch] -o out.md      # 对比表 + 预估vs实际
```

---

## 8. 里程碑

| 版本 | 范围 |
|---|---|
| **v0.1**（最小闭环） | 管道化实验类（cfg_cls + pipeline + Stage 契约 + transform 缓存）、BaseRun 骨架（本地后端 + 自适应并行）、step 测量窗口滚动预估（diversity-first + log-log surrogate）、check/report、校准回填。不做：MLflow 导出、Slurm/云、通知通道、早停分布学习、自动代码 hash |
| v0.2 | 早停分布真实学习、surrogate 升级（更强模型）、resume 强化、报告模板化、通知通道 |
| v0.3 | Slurm/云后端、Optuna / MLflow 集成、分批预算排期（OQ-7） |

---

## 9. 待决策问题清单（OQ）

- **OQ-1** 接入约定 A（进程）与 B（库模式）哪个为主？两者都要？
- **OQ-2** 你现有训练代码的形态：入口是什么？metric 现在怎么输出？（决定 §4.1 是否需要 stdout 解析兼容）
- **OQ-3** workload 族由用户在 BatchSpec 显式声明（`group_by`），可接受吗？还是希望自动聚类？
- **OQ-4** 早停先验 0.3、早停 run 耗时 0.15× 全量——符合你实验的实际吗？
- **OQ-5** 可行性判定粒度：整批一个 verdict 即可，还是需要 per-维度敏感性（"去掉 model=TimesNet 可提前 Y 小时"）？
- **OQ-6** 跨族元模型（参数量×数据量×batch → 单位时间）是否进 v0.1？（建议不进）
- **OQ-7** 是否需要"分批预算"（每周/每月 100 卡时，库自动排期）？
- **OQ-8** 命名 "expman" 是否 OK？
- **OQ-9** 记录库需要支持多用户/只读分享（实验室场景）吗？
- **OQ-10** 运行环境：WSL 内即可，还是 Windows 原生也要能跑？

---

# 附录：来自真实工作流访谈的产出（2026-09-06）

> 访谈对象：文档作者本人。事实基线：ini 配置文件；本地小验证 → git push → AutoDL 大批量；本地/云端配置差异（梯度累积等）靠手改文件导致 git 冲突；跑挂主因 HF 下载网络失败；已有批量级跳过（无 epoch 级续跑）；结果靠一次性 parse 脚本，历史结果不可检索；监控 = 刷 AutoDL jupyterlab，无总视窗。

## §10 用例场景（User Stories）

**US-1 本地调通 → 云端放大（日常主流程）**
27 个 run（3 模型 × 3 缺失率 × 3 seed）。用户写一个 batch.yaml；环境差异（batch_size / grad_accum / device）声明在环境覆盖层，**不进 git、不手改文件**。本地跑 2-3 个验证 → git push → AutoDL 上 `expman run --resume` 自动注入云端覆盖。
验收：全程零次手动修改配置文件；无 restore/pull/再改循环。

**US-2 断连/关机后重启续跑**
跑到 17/27 实例断。`expman run --resume`：1-16 号 run 跳过（现状已有），**17 号从 step 41,200（epoch 137/200）续跑**（新需求）。expman 发现 ckpt 即自动传 resume 标志，用户不需记得"上次跑到哪"。
验收：重启后人工介入 ≤ 1 条命令。

**US-3 网络失败自动重试**
HF 下载失败不再杀死 run：网络类错误自动重试 3 次；模型权重走共享缓存目录（HF_HOME 指向实例持久盘，预下载一次全家复用）。
验收：网络抖动场景下无人工干预，run 不白跑。

**US-4 结果考古（论文期硬需求）**
两个月后查"SAITS × PhysioNet × 缺失率 0.3 的 RMSE 与当时配置"：`expman report --filter "model=SAITS and missing=0.3"` 出表；导出 md/CSV/LaTeX 表格、baseline diff 列、绘图数据。
验收：答得出历史实验的最终指标 + 复现所需 config；论文表格粘贴即用。

**US-5 总视窗与完成通知**
`expman status`：`[17/27] SAITS_m0.3_s1 · running · step 41,200/60,000 (ep 137/200) · 本 run 4.2h 已用 · ETA 1.8h · 整批 ETA 2026-09-08 03:00`。整批结束推送手机（AutoDL 端轻量推送 API；本地可复用 Hermes 微信通道）。
验收：任何时候能回答"跑到哪了、还剩多久"；结束时收到通知。

**US-6 自适应并行（同卡多实验，省卡时）**
租 1 张 24G 卡跑 6-8G 的小实验，`device.concurrent: auto`。expman 先启动 1 个 run 观察显存峰值 → 峰值稳定且有余量再启动第 2 个 → 持续装箱；并发挤爆 OOM 的 run 自动放回池尾并降级该失败组合，不跳过不判死；独占仍 OOM 才报"配置不可行（需 ≥X GB 或换卡）"。
验收：无人工干预下实现同卡 2-3 run 并行（卡时成本下降）；无活锁（失败组合不反复重试）；真 OOM 最终判死而非无限重试。

**US-7 开工前整批预估 + 滚动收敛（时间评估主用例）**
27 个 run（3 模型 × 3 数据集 × 3 seed，各 200 epoch）。diversity-first 先跑 6 个覆盖不同模型 × 数据集的 run，各前 100 step 测量窗口 → 首个决策点输出：`整批预估 41h {P10 36h, P90 52h}，预算 72h → ✅ 可行`；若预算 24h → ❌ 不可行，并给量化削减建议（如"去掉 TimesNet 维度降至 29h"）——此刻只浪费 6×100 step 的训练量。运行中 `status` 显示预估滚动收敛：`整批预估 41h（已确认 62%，±4h）· [17/27]`。
验收：首个决策点不晚于前 6 个多样化 run 完成窗口；预估误差随进度单调下降；不可行时在浪费 <1% 总预算内止损并给量化建议。

**US-8 批量推理（无 epoch 的 workload）**
拿预训练模型对 5 个数据集 × 2 组采样设置做批量推理（10 个 run，`workload.type: inference`，每个 run 声明 `batches` = step 总数）。expman 用前 100 step（≥3s）测 s/step 与显存峰值，整批预估与可行性判定照常；`status` 显示 `[7/10] · step 34,000/50,000 · ETA 2.1h`；断连后 `--resume` 从已完成的 step 续跑。
验收：推理 run 全程无 epoch 概念即可估时/监控/续跑，与 train 共用同一套预估管线。

**US-9 异构参数空间快速收敛（diversity-first + surrogate）**
27 个 run 混合大/小模型（单位时间差 10×）。顺序跑则前 3 个同类小模型会导致整批预估严重低估；diversity-first 先跑 6 个覆盖不同模型 × 规模的 run → 决策点预估误差 <±20%；surrogate 预填剩余 run 的 ETA，随实测替换收敛。
验收：决策点（前 6 个多样化 run）整批预估相对误差 <20%（留一 CV 佐证）；surrogate 只影响未测 run 的预填，已测 run 的最终记录不受影响。

**US-10 管道化与预处理缓存**
27 个 run 共享 3 种 missing_rate。首个 run 执行 LoadRaw→InjectMissing→Normalize→Window 后产物按 cfg_hash 落盘；后续同配置 run 命中缓存直接进 Train。改预处理代码后 stage `version` +1，缓存自动失效重建。
验收：同配置预处理不重复执行（日志可见 cache hit）；批量预估中 prep 时间只计唯一配置；改代码且 version+1 后不静默复用旧产物。

## §11 访谈发现 → 设计修正提案（M1-M14，⏳ 待确认 / ✅ 已合入）

| # | 发现 | 修正提案 | 影响范围 | 状态 |
|---|---|---|---|---|
| M1 | 配置是 ini，非 yaml | 接入层"配置格式无关"：模板文件 + override 注入；必须能读 ini | §4.1 | ⏳ |
| M2 | 本地/云端差异手改文件 → git 冲突地狱 | **配置三层分离**：实验类 cfg（进 git）/ 环境层（按机器 profile 注入，永不进 git）/ BatchSpec dims 覆盖；v0.7 实验类集中配置落实 | §3 §4 | ✅ 已合入 v0.7 |
| M3 | 大批量在 AutoDL，显卡不同 | probe 必须在目标机型跑；(workload, 机型) 键控缓存 | §5 | ⏳ |
| M4 | 要 epoch 级续跑而非仅批量跳过 | checkpoint 约定进接入层：每 N epoch 落盘 + 自动传 resume 标志 | §4.1 §4.4 | ⏳ 改动最大 |
| M5 | 主失败模式 = HF 下载网络错误 | 失败分类：网络类自动重试 vs 代码类停报；模型缓存目录约定 | §4.4 | ⏳ |
| M6 | 结果要"普适分析系统"：论文表格/baseline diff/绘图 | Recorder/Reporter 升格为查询分析层：SQLite 底座 + 查询/导出/绘图数据为核心功能 | §4.5 §6 | ⏳ |
| M7 | 无总视窗、无完成通知 | status 输出 run 位置 + 双 ETA（本 run/整批）；完成通知 hook | §7 | ⏳ |
| M8 | AI 写代码致配置层次混乱 | 实验类集中配置 + 管道阶段边界（v0.7 落实）；cfg 校验器留 v0.1 实现 | §3 §4 | ✅ 已合入 v0.7 |
| M9 | 单卡多实验并行（显存感知、动态扩容、OOM 放回池） | 自适应并行调度器（US-6 / §4.4.1）；拍板默认：探峰窗口 3 epoch、margin 10%、α 校准学习 | §4.4 §6 §3 | ✅ 已合入 v0.3 |
| M10 | 批量总耗时要覆盖"不同数据集/模型差异"；思路=每 run 先测几 epoch | 时间评估改为测量窗口 + 滚动预估（US-7 / §5.2 重写）；双模式 every_run（默认）/ per_family | §5 §3 | ✅ 已合入 v0.4 |
| M11 | 推理实验（预训练模型批量推理）无 epoch 概念，原预估管线不兼容 | workload 泛化：T = total_steps × s/step；type=train\|inference（v0.6 起 step 为唯一单位） | §3 §5 | ✅ 已合入 v0.5，v0.6 演进 |
| M12 | 用部分参数组合的运行时间估计其他组合 | diversity-first 启动序 + cost surrogate 预填（留一 CV 质量监控、实测永远优先） | §5.2 §5.3 | ✅ 已合入 v0.5 |
| M13 | 对接方式定为基类（规范化代码撰写）；时间单位定为 step | BaseRun 基类双轨（轨道1 默认骨架+逃生舱；轨道2 纯黑盒兼容）；step 为唯一单位（T = total_steps × s/step） | §4.1 §3 §5 | ✅ 已合入 v0.6 |
| M14 | 数据处理/训练结构化为管道（模块只传数据）；配置集中于实验类 | 管道化实验类（cfg_cls + pipeline + Stage 契约）；transform 缓存按 cfg_hash；批量 prep 唯一化 | §4.1 §3 §5 | ✅ 已合入 v0.7 |

**定位修正（访谈结论）**：北极星 = 接入一次后，新实验只需"写一个 batch.yaml + 一行 run"，环境差异/断点/估时/结果查询不再操心。此定位同时强化 D1（自建必要性）：MLflow+Hydra 拼装解决不了 M2/M3/M4 这类"研究者个人工作流"问题。
