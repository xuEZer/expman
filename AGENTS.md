# 项目开发规范

- 每次执行 `git commit` 前，必须运行并通过 `ruff check .` 和 `ruff format --check .`，包括仅修改文档的提交。
- 修改 Python 代码后重新执行上述检查。需要修复时使用 `ruff check --fix .` 和 `ruff format .`，检查差异后重新验证。
- 安装开发依赖及提交 hook：`python -m pip install -e '.[dev]'`，然后 `pre-commit install --install-hooks`。
- 正常提交必须经过仓库的 pre-commit 检查，不使用 `--no-verify` 或 `SKIP` 绕过。
- 代码行为发生变化时运行相关测试；提交规范与版本管理遵循 `CONTRIBUTING.md`。
