# 开发与版本管理

## 分支与提交

- `main` 保存经过验证的工作版本。
- 新功能使用 `feat/<name>`，修复使用 `fix/<name>`，文档维护使用 `docs/<name>`。
- 一次提交围绕一个完整目的，代码、必要测试与对应文档一起提交。
- 提交消息采用 Conventional Commits：`feat:`、`fix:`、`docs:`、`test:`、`chore:`、`refactor:`。例如 `feat: add sequential pipeline execution`。
- **每次提交前必须通过 `ruff check .` 和 `ruff format --check .`，包括仅修改文档的提交。** 修复后重新检查，暂存确认过的改动，再提交。
- 合并前检查变更、运行测试和示例，确认工作区没有实验数据、密钥、构建产物或本地环境文件。

项目使用 uv 管理唯一的锁定环境，其中只有运行依赖（NumPy、PyTorch、PyYAML）。Ruff 和 pre-commit 是开发工具，单独安装，不进入包依赖，也不设 dev extra：

```bash
uv sync
python -m pip install pre-commit ruff
pre-commit install --install-hooks
```

每次新克隆仓库或重建环境后，都需要重新安装 hook。

提交前检查：

```bash
ruff check .
ruff format --check .
uv run python -m unittest discover -s tests -v
uv run python examples/basic_pipeline.py
git diff --check
git status --short
```

需要自动修复时，执行 `ruff check --fix .` 和 `ruff format .`，然后检查差异并重新运行上述检查。

`.pre-commit-config.yaml` 为每次提交运行全项目 Ruff lint 和格式检查，任一失败都会阻止提交。hook 只检查，不自动修改或暂存文件；pre-commit 会临时隐藏已跟踪文件的未暂存改动，以检查本次提交对应的内容。不要使用 `--no-verify` 或 `SKIP` 绕过检查。

Ruff 版本固定在两处：`.pre-commit-config.yaml` 的 `additional_dependencies`（本地 hook）和 `.github/workflows/ci.yml`（CI）。升级时一起改。

项目附带 GitHub Actions 配置，在推送或创建 PR 后执行 Ruff 检查，以及 Python 3.10–3.14 的安装、测试和示例检查。远端分支保护需要在托管平台另行配置。

## 版本

版本号集中维护在 `pyproject.toml`，格式为 `MAJOR.MINOR.PATCH`，对应 Git 附注标签 `vMAJOR.MINOR.PATCH`。

- 当前处于 `0.x` 开发期，公共接口仍在演进。
- `0.x` 的接口不兼容调整通过提高次版本号发布，并在 CHANGELOG 中提供迁移说明。
- 补丁版本用于兼容修复；新增兼容功能提高次版本号。
- 达到 `1.0.0` 后，接口不兼容变更提高主版本号。

发布步骤：

1. 更新项目版本及 `CHANGELOG.md`，保持 README 的当前版本一致。
2. 完成 Ruff lint、格式检查、测试、示例及安装检查，审核 Git 差异。
3. 提交版本变更，再创建附注标签，例如 `git tag -a v0.1.0 -m "expman 0.1.0"`。
4. 在配置好远端并获得发布授权后推送分支和标签；包分发另行安排。

## 代码约定

- 公共接口提供类型注解和文档；Stage 子类覆盖 `process()`，统一通过 `run()` 调用。
- 对业务规则、故障行为和对外契约做测试，避免只测试私有实现细节。
- 核心执行层仅依赖标准库；配置模块使用 PyYAML，存储和监控集成通过 Recorder 扩展。
- 新增功能同步修改设计文档及 CHANGELOG，明确已实现能力与后续规划。

## 完整后端验证

需要验证完整功能时，在可访问实际 NVIDIA GPU 的环境运行：

```bash
uv run python scripts/test_full.py
```

该入口执行全部测试，并将任何 skip 视为验证未完成（非零退出码）。缺少依赖、GPU 不可访问或显存不足时，修复环境后重跑；不要删除测试或取消跳过条件来获得通过。普通 unittest 命令仍可用于最小依赖环境的兼容性测试，其结果不能替代完整后端验证。

当前开发机运行时也保持项目源码优先：

```bash
PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=GPU-23e616ac-ab14-91bd-71e2-ac9a8887337b \
uv run python scripts/test_full.py
```

上面的 GPU UUID 仅适用于当前开发机；其他机器使用自己的完整测试环境。沙箱阻止 GPU 访问时，需要在获准访问 GPU 的执行环境运行。
