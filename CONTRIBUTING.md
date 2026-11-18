# 开发与验证

## 环境与提交检查

项目使用 uv。包依赖为 PyYAML、NumPy 和 PyTorch，开发依赖组包含 pytest、Pyright 和 pre-commit。Ruff 用于检查源码。安装环境及本地提交 hook：

```bash
uv sync
python -m pip install pre-commit ruff
pre-commit install --install-hooks
```

修改 Python 代码后、每次提交前都运行以下检查。仅修改文档的提交也执行同样的检查。若项目环境已单独安装与本机驱动匹配的 PyTorch，手动检查时可设置 `UV_NO_SYNC=1`，让 uv 使用当前环境，与提交 hook 一致。

```bash
ruff check .
ruff format --check .
uv run pyright
uv run pytest
```

需要自动修复时运行 `ruff check --fix .` 和 `ruff format .`，检查差异后重新验证。提交 hook 还会运行 `uv lock --check`。GitHub Actions 在推送和拉取请求时运行 Ruff、Pyright，并在 Python 3.12、3.13、3.14 上运行 CPU 测试。

CI 的真卡测试使用 `expman-gpu` 标签。该标签用于每个作业独立分配、作业结束即销毁的 Linux x64 runner，须提供 NVIDIA GPU、MPS，以及与锁文件中 PyTorch 兼容的驱动。没有匹配的 runner 时，作业会排队等待。

## 分支与提交

- `main` 保存通过验证的代码。功能、修复和文档分支分别使用 `feat/<name>`、`fix/<name>` 和 `docs/<name>`。
- 一次提交围绕一个目的，一起提交实现、相应的行为测试和接口说明。提交消息采用 Conventional Commits，例如 `fix: preserve checkpoints on retry`。
- 提交前用 `git diff --check` 和 `git status --short` 核对改动。正常提交须经过仓库的 pre-commit 检查。

## 代码约定

- 为公共接口提供类型注解和文档。业务阶段实现 `Stage.process`，按需声明 `config_dependencies`。
- 为使用者可观察的行为与故障恢复写测试。修改行为时同步核对 README 中的示例和架构说明。

## 真实 GPU 验证

pytest 和提交 hook 默认运行 `tests/cpu/` 和 `tests/gpu/`。后者需要 Linux、NVIDIA 驱动、MPS，以及与驱动兼容的 CUDA 版 PyTorch。单独运行真卡测试可执行

```bash
uv run --no-sync pytest tests/gpu/
```

提交 hook 和上面的命令都使用已安装的项目环境，避免提交时自动替换 PyTorch。

测试检查主机剩余内存，并在 `CUDA_VISIBLE_DEVICES` 可见、空闲显存能容纳初始 worker 限额的物理卡中，选择 `nvidia-smi` 采样利用率最低的一张。正在运行其他计算任务的卡也参与排序。利用率相同时优先选空闲显存更多的卡，显存也相同时选物理编号较小的卡。

并发测试按 GPU UUID 加锁，在同一 Linux 网络命名空间中各自领取一张卡，直到本次真卡用例结束。测试输出选中的物理卡编号和 UUID。硬件条件不足或 CUDA/MPS 用例执行出错都会使测试和提交检查失败。
