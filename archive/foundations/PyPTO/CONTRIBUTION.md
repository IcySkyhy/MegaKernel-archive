# 贡献指南

本项目欢迎广大开发者体验并参与贡献，在参与社区贡献之前。请参见[cann-community](https://gitcode.com/cann/community)了解行为准则，进行CLA协议签署，了解源码仓的贡献流程。

开发者准备本地代码与提交PR时需要重点关注如下几点：

1. 提交PR时，请按照PR模板仔细填写本次PR的业务背景、目的、方案等信息。
2. 若您的修改不是简单的bug修复，而是涉及到新增特性、新增接口、新增配置参数或者修改代码流程等，请务必先通过Issue进行方案讨论，以避免您的代码被拒绝合入。若您不确定本次修改是否可被归为“简单的bug修复”，亦可通过提交Issue进行方案讨论。


开发者贡献场景主要包括：

- Bug修复

  如果您在本项目中发现了某些Bug，希望对其进行修复，欢迎您新建Issue进行反馈和跟踪处理。

  您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Bug-Report|缺陷反馈` 类Issue对Bug进行描述，然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您进行处理。

- 代码优化

  如果您对本项目中某些API实现有泛化性增强/性能优化思路，希望着手实现这些优化点，欢迎您对API进行优化贡献。

  您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Requirement|需求建议` 类Issue对优化点进行说明，并提供您的设计方案，
  然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您进行跟踪优化。

- 文档纠错

  如果您在本项目中发现某些文档描述错误，欢迎您新建Issue进行反馈和修复。

  您可以按照[提交Issue/处理Issue任务](https://gitcode.com/cann/community#提交Issue处理Issue任务)指引新建 `Documentation|文档反馈` 类Issue指出对应文档的问题，然后在评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您纠正对应文档描述。

- 帮助解决他人Issue

  如果社区中他人遇到的问题您有合适的解决方法，欢迎您在Issue中发表评论交流，帮助他人解决问题和痛点，共同优化易用性。

  如果对应Issue需要进行代码修改，您可以在Issue评论框中输入“/assign”或“/assign @yourself”，将该Issue分配给您，跟踪协助解决问题。

---

## 代码规范与 Pre-commit 检查

本项目使用 [pre-commit](https://pre-commit.com/) 框架在提交前自动执行代码风格检查与格式化，确保所有贡献代码遵循统一的编码规范。开发者在提交代码前**必须**通过 pre-commit 检查。

### 安装

```bash
pip install pre-commit
pre-commit install
```

安装完成后，每次 `git commit` 都会自动触发检查。如果检查未通过，提交将被拦截。

### 检查项一览

检查配置位于 `.pre-commit-config.yaml`，主要包括：

| 检查项 | 工具 | 说明 |
|--------|------|------|
| 尾部空格 / 文件末尾换行 | pre-commit-hooks | 清理行尾空格，确保文件以换行符结尾 |
| YAML / JSON 合法性 | pre-commit-hooks | 校验配置文件语法 |
| 大文件 / 私钥 / 合并冲突检测 | pre-commit-hooks | 防止误提交二进制文件和密钥，检查合并冲突标记 |
| C++ 代码格式化 | clang-format (v18) | 按 `.clang-format` 规则格式化 C/C++/ASC 文件 |
| Python 代码检查 | ruff check (v0.14) | 静态检查 E/W/F/I/N 规则族，自动修复可修复项 |
| Python 切片冒号空格 | local 脚本 | 检查切片冒号两侧空格（补充 ruff 覆盖盲区） |
| 拼写检查 | codespell | 检查常见拼写错误 |

### Python 代码规范（ruff）

ruff 配置位于 `pyproject.toml` 的 `[tool.ruff]` 段，启用的规则族：

- **E / W** — pycodestyle 错误与警告（如 E711 比较运算、E741 歧义变量名）
- **F** — pyflakes（如 F401 未使用导入、F841 未使用变量、F821 未定义名称）
- **I** — isort 导入排序
- **N** — pep8-naming 命名规范（如 N802 函数名、N806 变量名、N818 异常名）

已忽略 `E501`（行长度），行宽上限为 **120** 字符。

### 手动运行检查

```bash
# 对所有文件运行全部检查
pre-commit run --all-files

# 仅运行 ruff 检查
pre-commit run ruff-check --all-files

# 仅运行 C++ 代码格式化检查
pre-commit run clang-format --all-files

# 仅运行切片冒号空格检查
pre-commit run slice-colon-spacing --all-files

# 仅运行拼写检查
pre-commit run codespell --all-files
```
