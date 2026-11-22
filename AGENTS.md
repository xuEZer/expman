# 项目开发规范

- 不要自行运行测试。执行 `git commit` 时由 pre-commit 自动运行测试，以及 Ruff、Pyright 和锁文件检查。
- hook 报错时修复问题、检查差异，再次提交以触发验证。需要自动修复时使用 `ruff check --fix .` 和 `ruff format .`。
- 安装提交 hook：`python -m pip install pre-commit ruff`，然后 `pre-commit install --install-hooks`。这两个是开发工具，不是包的依赖：项目只发布一个包，不维护 dev extra。
- 正常提交必须经过仓库的 pre-commit 检查，不使用 `--no-verify` 或 `SKIP` 绕过。
- 代码行为发生变化时同步维护相关测试、文档和注释。提交规范遵循 `CONTRIBUTING.md`。
