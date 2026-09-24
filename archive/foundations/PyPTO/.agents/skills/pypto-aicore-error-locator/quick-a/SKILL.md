---
name: pypto-aicore-error-locator-quick-a
description: 快捷模式 A — 已知 CCE 文件路径，跳过步骤 1-3，直接执行二分查找定位问题代码行并映射到源码。当用户提供了 CCE 文件路径但未提供问题行号时触发。
---

# 快捷模式 A：已知 CCE 文件 → 二分查找 + 映射

用户已通过其他方式定位到问题 CCE 文件，本模式跳过前置步骤，直接从二分查找开始。

## 前置确认

从用户输入中提取：
- **cce_file**：问题 CCE 文件路径（必须）
- **test_cmd**：触发 aicore error 的测试命令（必须，二分查找需要运行测试验证）
- **run_path**：运行的目录路径（默认当前目录）
- **Pypto_Test 框架**：根据用户上下文判断是否需要 `--use-pypto-test-framework`

**⚠️ 重要提示**: 将 bash 运行命令超时时间设置为 1800000ms

## 步骤 1：启用追踪日志

启用 trace 日志，使测试运行时将 program.json 生成到固定路径 (`run_path/output/`)，供后续源码映射使用。

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/setup_trace_logs.py
```

**⚠️ 重要提示**：
- 若脚本退出码为 0：trace 已启用（或已为目标状态），**继续执行后续步骤**
- 若脚本退出码为 1：未找到已安装的 tile_fwk_config.json，**停止执行后续步骤**

## 步骤 2：验证 CCE 文件

注释 CCE 文件中所有可注释的代码行，运行测试，确认此文件确实是问题文件。

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/verify_cce_file.py <cce_file> <test_cmd> <run_path> [--use-pypto-test-framework]
```

**⚠️ 重要提示**：
- 若脚本退出码为 0：此文件是问题文件，**继续执行后续步骤**
- 若脚本退出码为 1：此文件不是问题文件，**停止执行后续步骤**

## 步骤 3：准备二分查找 — 确定错误范围 + 获取初始范围

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/setup_binary_search.py <cce_file> <test_cmd> <run_path> [--use-pypto-test-framework]
```

**⚠️ 重要提示**:
- 记录输出的 `ERROR_IN_T`、`LEFT`、`RIGHT` 三个值供步骤 4 使用
- 若 `ERROR_IN_T=True`：后续二分查找范围仅限 T 操作行；若为 `False`：范围为除同步行外的所有行

## 步骤 4：执行二分查找迭代

**⚠️ 此步骤需要多次迭代执行**：每一步执行一条命令，根据输出结果更新 left/right 值后继续迭代，直至找到问题行。

根据步骤 3 的 `LEFT` 和 `RIGHT` 值，执行当前迭代：

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/binary_search_iteration.py <cce_file> <test_cmd> <run_path> <left> <right> <error_in_t> [--use-pypto-test-framework]
```

记录输出的 `NEXT_LEFT` 和 `NEXT_RIGHT` 值。

**判断逻辑**:
- 如果 `NEXT_LEFT` 等于 `NEXT_RIGHT`，则已找到问题行（输出 `FOUND <problem_line>`），记录 `problem_line` 值用于步骤 5
- 否则，使用新的 `NEXT_LEFT` 和 `NEXT_RIGHT` 作为下一轮的 `left` 和 `right`，重复执行此步骤

**⚠️ 重要提示**:
- `<error_in_t>` 为步骤 3 输出的 `ERROR_IN_T` 的值（`True` 或 `False`），必须传入字面量 `True`/`False`，不可传入变量名
- 每次迭代仅执行一条命令（受超时限制），需要在多轮中逐步收敛
- 若未定位到问题代码行（left > right 异常），请说明原因，**停止执行后续步骤**

## 步骤 5：CCE 行映射到前端源文件

二分查找的测试运行会在 `run_path/output/` 下生成 program.json。在该目录中查找最新的 `output_*` 子目录获取 program.json 路径：

```bash
find <run_path>/output -name "program.json" -path "*/output_*/*" | head -1
```

使用获取到的 program.json 路径进行源码映射：

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/locate_source_line.py <cce_file> <program_json_path> <problem_line>
```

**参数说明**:
- `<cce_file>`：用户提供的 CCE 文件路径
- `<program_json_path>`：从 `run_path/output/` 下找到的 program.json 路径
- `<problem_line>`：步骤 4 输出的问题代码行号

## 步骤 6：输出最终结果

汇总输出以下信息：
- CCE 文件路径
- 问题代码行号
- 问题代码内容
- 前端源代码文件路径（如果步骤 5 映射成功）
- 前端源代码行号（如果步骤 5 映射成功）
- 前端源代码内容（如果步骤 5 映射成功）

## 恢复初始状态

定位完成后，恢复步骤 1 修改的 tile_fwk_config.json：

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/restore_initial_state.py --pypto-path <pypto_path>
```
