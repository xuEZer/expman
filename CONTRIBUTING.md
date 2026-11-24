# 开发与验证

## 环境与提交检查

项目使用 uv。包依赖为 PyYAML、NumPy 和 PyTorch，开发依赖组包含 pytest、Pyright 和 pre-commit。Ruff 用于检查源码。安装环境及本地提交 hook：

```bash
uv sync
python -m pip install pre-commit ruff
pre-commit install --install-hooks
```

执行 `git commit` 时，pre-commit 自动运行 Ruff 检查和格式检查、Pyright、pytest 和锁文件检查。仅修改文档的提交也执行同样的检查。

需要自动修复时运行 `ruff check --fix .` 和 `ruff format .`，检查差异后再次提交。GitHub Actions 在推送和拉取请求时运行 Ruff、Pyright，并在 Python 3.12、3.13、3.14 上运行 CPU 测试。

CI 的真卡测试使用 `expman-gpu` 标签。该标签用于每个作业独立分配、作业结束即销毁的 Linux x64 runner，须提供 NVIDIA GPU、MPS，以及与锁文件中 PyTorch 兼容的驱动。没有匹配的 runner 时，作业会排队等待。

## 分支与提交

- 本项目由一人维护。日常开发直接在 `main` 上进行，检查通过后提交。
- 一次提交围绕一个目的，一起提交实现、相应的行为测试和接口说明。提交消息采用 Conventional Commits，例如 `fix: preserve checkpoints on retry`。
- 提交前用 `git diff --check` 和 `git status --short` 核对改动。正常提交须经过仓库的 pre-commit 检查。

## 代码约定

- 为公共接口提供类型注解和文档。业务阶段实现 `Stage.process`，按需声明 `config_dependencies`。
- 为使用者可观察的行为与故障恢复写测试。修改行为时同步核对 README 中的示例和架构说明。

## 真实 GPU 验证

提交 hook 自动运行 `tests/cases/` 下的全部用例。`cpu/` 下的用例可在 CPU 上运行，`gpu/` 下的用例需要 Linux、NVIDIA 驱动、MPS 和兼容的 CUDA 版 PyTorch。两个目录各自通过 `conftest.py` 管理运行环境。创建、恢复和等待批次的函数以及真卡预留工具位于 `tests/helpers/`。

提交 hook 中的 Pyright 和 pytest 均使用 `uv run --no-sync`，保留项目环境中已安装的 PyTorch。

测试检查主机剩余内存，并在 `CUDA_VISIBLE_DEVICES` 可见、空闲显存能容纳初始 worker 限额的物理卡中，选择 `nvidia-smi` 采样利用率最低的一张。正在运行其他计算任务的卡也参与排序。利用率相同时优先选空闲显存更多的卡，显存也相同时选物理编号较小的卡。

第一次运行真卡用例时，pytest 按上述顺序尝试预留一张卡，并输出物理卡编号和 UUID。同一 Linux 网络命名空间内的并发 pytest 进程按 UUID 互斥，每个进程持有预留直到自己的测试会话结束。所有候选卡都已被其他测试预留、硬件条件不足或 CUDA/MPS 用例执行出错时，检查会失败。
