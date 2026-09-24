---
name: pypto-aicore-error-locator
description: 定位测试案例中出现 aicore error 时的问题 CCE 文件和问题代码行。当用户说"aicore error"、"定位 aicore error 的原因"、"帮我定位 aicore error 报错"时使用此技能。也适用于用户直接提供 CCE 文件希望定位问题代码行或映射到源码的场景。
---

# AICore Error 定位器

此技能是总入口，根据用户输入自动分流到对应路径。

## 路径选择（总入口）

根据用户提供的输入，直接分流：

| 用户输入特征 | 分流到 | 说明 |
|---|---|---|
| 提供了 CCE 文件路径 + 问题代码行号 | [quick-b](quick-b/SKILL.md) | 直达源码映射 |
| 提供了 CCE 文件路径（无行号） | [quick-a](quick-a/SKILL.md) | 跳过步骤 1-3，从二分查找开始 |
| 未提供 CCE 文件 | **完整流程**（下方） | 执行标准步骤 1-6 |

> 判断依据：用户消息中包含 CCE 文件路径（如 `.cpp` 文件），以及是否同时提供了行号。

---

## 完整流程

1. 初始测试 — 问题复现，检查是否为该脚本可适用的 aicore error 场景
2. 排除 machine 框架调度问题 — 判断问题在 kernel 代码还是 machine 调度框架
3. 定位问题 CCE 文件 — 通过单个脚本完成追踪日志启用、重编、测试、日志分析和 CCE 文件验证
4. 二分 CCE 文件找到问题代码行
    4.1 准备二分查找 — 确定错误范围 + 获取初始范围
    4.2 二分查找迭代
5. 问题代码行映射到前端代码
    5.1 CCE 行映射到前端源文件
    5.2 输出最终结果

### Pypto_Test 框架适配

**重要**：如果用户明确使用 Pypto_Test 框架，所有脚本调用必须追加 `--use-pypto-test-framework` 参数。判断依据（**不区分大小写、不区分下划线 `_` 和中划线 `-`**）：
- 用户提到 "Pypto_Test"、"pypto test"、"PYTO_TEST"、"pypto-test" 等任意变体
- 测试命令中包含 pypto test 框架组件的导入或调用
- 用户明确说"使用框架"、"框架模式"

---

## 步骤 1：初始化默认信息与初始测试

使用以下默认值直接初始化：

- **pypto_path**: 当前工作目录（即 pypto 项目根目录）
- **device_log_path**: `{pypto_path}/device_log`
- **run_path**: `{pypto_path}`
- **test_cmd**: 从用户输入中提取（用户触发此技能时提供的测试命令或问题描述中包含的运行命令）

**⚠️ 重要提示**: 将 bash 运行命令超时时间设置为 1800000ms

### 初始测试 - 确认 aicore error 可复现

**在修改任何代码之前**，先运行一次测试，确认 aicore error 确实存在：

```bash
cd <run_path> && export ASCEND_GLOBAL_LOG_LEVEL=0 && <test_cmd>
cd -
```

**⚠️ 重要提示**：检查打屏输出，必须出现 `aicore error`。
- **如果未出现 aicore error**：说明该问题不适用此定位方法（可能已修复或测试命令不正确），**立即停止执行后续步骤**！
- **如果出现 aicore error**：确认问题可复现，继续执行后续步骤。

---

## 步骤 2：排除 machine 框架调度问题

通过单个脚本判断 aicore error 源于 kernel 代码还是 machine 调度框架。该脚本合并了原来的 2.1-2.4 子步骤（定位 aicore_entry.h → 注释 CallSubFuncTask → 运行测试 → 恢复文件），**直接操作已安装的 pypto 包，无需重新编译安装**。

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/exclude_machine_framework.py \
  --test-cmd <test_cmd> \
  --run-path <run_path> \
  [--use-pypto-test-framework]
```

**⚠️ 重要提示**：
- 使用 bash 运行命令超时时间为 1800000ms
- 若脚本退出码为 0（注释后无 aicore error）：问题在 kernel 代码中，**继续执行后续步骤**
- 若脚本退出码为 1（注释后仍有 aicore error）：问题在 machine 框架调度，**停止执行后续步骤**
- 若脚本退出码为 2（定位 aicore_entry.h 或 CallSubFuncTask 失败）：**停止执行后续步骤**，手动排查环境

---

## 步骤 3：定位问题 CCE 文件

通过单个脚本完成 3.1-3.7 的全部操作：启用追踪日志 → 条件重编译 → 运行测试 → 分析日志 → 定位并验证 CCE 文件。脚本内部自动处理并行编译错误（`ld.lld: error: undefined`），自动重试。

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/locate_problem_cce.py \
  --pypto-path <pypto_path> \
  --test-cmd <test_cmd> \
  --run-path <run_path> \
  --device-log-path <device_log_path> \
  [--use-pypto-test-framework]
```

**⚠️ 重要提示**：
- 使用 bash 运行命令超时时间为 1800000ms
- 若脚本退出码为 0：成功定位，输出 `CCE_FILE=<path>` 和 `PROGRAM_JSON=<path>`，记录这两个路径供步骤 4、5 使用
- 若脚本退出码为 1：未找到问题 CCE 文件，**停止执行后续步骤**
- 若脚本退出码为 2：并行编译错误无法自动修复（parallel_compile 已为 1 但仍报错），**停止执行**

---

## 步骤 4：二分 CCE 文件找到问题代码行

通过迭代注释缩小范围，精确定位到导致 aicore error 的具体代码行。

### 4.1 准备二分查找 — 确定错误范围 + 获取初始范围

单个脚本合并 4.1 和 4.2：先注释所有 T 操作行并测试确定错误是否在 T 操作中，再根据结果计算二分查找的初始范围。

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/setup_binary_search.py <cce_file> <test_cmd> <run_path> [--use-pypto-test-framework]
```

**⚠️ 重要提示**:
- `cce_file` 为步骤 3 脚本输出的 `CCE_FILE` 值
- 记录输出的 `ERROR_IN_T`、`LEFT`、`RIGHT` 三个值供步骤 4.2 使用
- 若 `ERROR_IN_T=True`：后续二分查找范围仅限 T 操作行；若为 `False`：范围为除同步行外的所有行

### 4.2 执行二分查找迭代

**⚠️ 此步骤需要多次迭代执行**：每一步执行一条命令，根据输出结果更新 left/right 值后继续迭代，直至找到问题行。

根据步骤 4.1 的 `LEFT` 和 `RIGHT` 值，执行当前迭代：

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/binary_search_iteration.py <cce_file> <test_cmd> <run_path> <left> <right> <error_in_t> [--use-pypto-test-framework]
```

记录输出的 `NEXT_LEFT` 和 `NEXT_RIGHT` 值。

**判断逻辑**:
- 如果 `NEXT_LEFT` 等于 `NEXT_RIGHT`，则已找到问题行（输出 `FOUND <problem_line>`），记录 `problem_line` 值用于步骤 5
- 否则，使用新的 `NEXT_LEFT` 和 `NEXT_RIGHT` 作为下一轮的 `left` 和 `right`，重复执行此步骤

**⚠️ 重要提示**:
- `cce_file` 为步骤 3 脚本输出的 `CCE_FILE` 值
- `<error_in_t>` 为步骤 4.1 输出的 `ERROR_IN_T` 的值（`True` 或 `False`），必须传入字面量 `True`/`False`，不可传入变量名
- 每次迭代仅执行一条命令（受超时限制），需要在多轮中逐步收敛
- 若未定位到问题代码行（left > right 异常），请说明原因，**停止执行后续步骤**

---

## 步骤 5：问题代码行映射到前端代码

将 CCE 问题行映射回前端源文件，便于开发者定位修复。

### 5.1 CCE 行映射到前端源文件

使用以下命令将 CCE 问题代码行映射到前端源代码：

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/locate_source_line.py <cce_file> <program_json_path> <problem_line>
```

**参数说明**:
- `<cce_file>`: 步骤 3 脚本输出的 `CCE_FILE` 值
- `<program_json_path>`: 步骤 3 脚本输出的 `PROGRAM_JSON` 值
- `<problem_line>`: 步骤 4.3 输出的问题代码行号

**输出说明**:
- 若匹配成功，将输出前端源代码文件路径和行号
- 若完全无法匹配，将说明原因，常见场景包括：
  - **无法找到 funcHash 或无法解析 CCE 文件**：CCE 文件中缺少 `funcHash` 标记或行号越界，脚本输出 `✗ 无法映射到源代码`
  - **框架自动生成代码**：该代码行为框架自动生成（非前端用户编写的代码），无源码与之映射
  - **操作数数量不一致**：CCE 文件中的操作数与 `program.json` 中对应函数的操作数不匹配，脚本输出 `CCE 文件与 program.json 操作数不一致`。可能原因是 CCE 文件与 program.json 版本不一致，请重新运行测试生成新的 CCE 文件和 program.json 后重试

### 5.2 输出最终结果

汇总输出以下信息：
- 找到的 CCE 文件路径
- 问题代码行号
- 问题代码内容
- 前端源代码文件路径（如果步骤 5.1 映射成功）
- 前端源代码行号（如果步骤 5.1 映射成功）
- 前端源代码内容（如果步骤 5.1 映射成功）

---

## 恢复初始状态

定位完成后，使用单个脚本恢复步骤 3 修改的配置文件并重新编译安装 pypto：

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/restore_initial_state.py --pypto-path <pypto_path>
```

脚本自动完成：恢复 `tile_fwk_config.json` 和 `device_switch.h` 的 `.backup` 备份 → 重新编译安装 pypto。

---

## 关键注意事项

1. **fixed 模式**: 确保 `fixed` 模式启用以保持输出路径不变
2. **路径规范**: 所有路径必须使用绝对路径
3. **默认路径**: 步骤 1 使用默认值初始化，无需收集用户输入
4. **停止条件**: 遇到不适用的情况或定位失败时，立即停止执行并说明原因
5. **并行编译问题**: 步骤 3 脚本自动处理 `ld.lld: error: undefined` 错误（修改 `parallel_compile` 并重编），脚本内部完成重试；若退出码为 2 表示无法自动修复（parallel_compile 已为 1 仍报错），停止执行
6. **步骤 2 无需重编**: 步骤 2 直接修改安装路径下的 `aicore_entry.h`，无需重新编译安装 pypto
7. **步骤 4.2 迭代执行**: 二分查找因单次命令有超时限制，分解为多轮迭代执行，每轮执行一条命令
