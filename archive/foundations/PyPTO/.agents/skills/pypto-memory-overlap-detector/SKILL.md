---
name: pypto-memory-overlap-detector
description: PyPTO MACHINE 内存重叠检测与修复技能。通过系统化流程检测和修复 workspace 内存重叠、内存管理策略问题导致的精度异常。当怀疑精度问题由内存重叠、workspace 不足、内存管理异常引起时使用此技能。触发词：内存重叠、内存重叠检测、workspace问题、内存管理异常、内存复用错误。
---

# PyPTO MACHINE 内存重叠检测与修复技能

此技能用于检测并尝试修复 PyPTO MACHINE 内存重叠问题。

---

## 工作流程概述

1. 初始化默认信息
2. 内存重叠检测准备
3. 执行内存重叠检测
4. 若存在内存重叠，尝试修复
5. 结果解读与总结

---

## 步骤 1：初始化默认信息

使用以下默认值直接初始化：

- **pypto_path**: 当前工作目录（即 pypto 项目根目录）
- **device_log**: `{pypto_path}/device_log`
- **run_path**: `{pypto_path}`
- **test_cmd**: 从用户输入中提取（用户触发此技能时提供的测试命令或问题描述中包含的运行命令）

---

**⚠️ 重要提示**: 将bash运行命令超时时间设置为1800000ms

## 步骤 2：内存重叠检测准备

**操作步骤**：

### 2.1 打开 Operation 信息 Dump 开关

修改 `framework/src/machine/utils/device_switch.h`：

- 设置 `#define ENABLE_DUMP_OPERATION` 为 `1`

### 2.2 重新编译 PyPTO whl 包并安装

```bash
cd pypto_path && python3 build_ci.py -f python3 --disable_auto_execute
pip install build_out/pypto*.whl --force --no-deps
cd -
```

### 2.3 使用脚本添加 debug_options

使用脚本自动向测试文件添加 `debug_options`：

```bash
python3 .agents/skills/pypto-memory-overlap-detector/scripts/add_debug_options.py add <test_file_path>
```

**脚本参数说明**:
- `add`: 添加 debug_options
- `test_file_path`: 测试用例文件路径

### 2.4 清理日志并运行测试

```bash
rm -rf device_log/* && cd run_path && export ASCEND_GLOBAL_LOG_LEVEL=1 && export ASCEND_PROCESS_LOG_PATH=device_log && test_cmd
cd -
```

---

## 步骤 3：执行内存重叠检测

### 3.1 使用脚本获取检测参数

运行脚本自动获取 `-d` 和 `-t` 参数：

```bash
python3 .agents/skills/pypto-memory-overlap-detector/scripts/get_memory_check_paths.py device_log run_path/output
```

**脚本参数说明**:
- `device_log`: 落盘日志根目录
- `run_path/output`: 运行目录中的 output **根目录**（**注意**：不要传递具体的 output_xxx 子目录，脚本会自动查找最新的子目录）

**示例**：
```bash
# 正确 ✅
python3 .agents/skills/.../get_memory_check_paths.py /path/to/wk /path/to/output

# 错误 ❌（不要传递具体的 output_xxx 子目录）
python3 .agents/skills/.../get_memory_check_paths.py /path/to/wk /path/to/output/output_20260401_165509
```

### 3.2 执行内存重叠检测

使用脚本输出的命令执行检测：

```bash
python3 tools/schema/schema_memory_check.py -d <device_log_dir> -t <dyn_topo_file>
```

### 3.3 恢复测试用例文件

使用脚本恢复被修改的测试用例文件并删除备份：

```bash
python3 .agents/skills/pypto-memory-overlap-detector/scripts/add_debug_options.py restore <test_file_path>
```

### 3.4 恢复 PyPTO 源代码

修改 `framework/src/machine/utils/device_switch.h`：

- 设置 `#define ENABLE_DUMP_OPERATION` 为 `0`

---

## 步骤 4：若存在内存重叠，尝试修复

**⚠️ 重要**：只有步骤 3 检测到内存重叠问题时，才执行本步骤。

### 4.1 扩大 workspace 大小

**问题现象**：检测到内存重叠，怀疑 workspace 大小不足

**操作步骤**：

#### 4.1.1 修改代码

修改文件：`python/pypto/frontend/parser/entry.py`

```python
# 原始代码
workspace_tensor = torch.empty(workspace_size, dtype=torch.uint8, device=device)

# 修改为：扩大 10 倍
workspace_tensor = torch.empty(workspace_size * 10, dtype=torch.uint8, device=device)
```

#### 4.1.2 重新编译并安装

```bash
cd pypto_path && python3 build_ci.py -f python3 --disable_auto_execute
pip install build_out/pypto*.whl --force --no-deps
cd -
```

#### 4.1.3 运行测试验证

```bash
cd run_path && test_cmd
cd -
```

**判断标准**：
- 如果问题不复现 → workspace 大小计算问题，需进一步分析计算逻辑
- 如果问题仍然存在 → 继续下一步排查

#### 4.1.4 恢复原代码

修改文件：`python/pypto/frontend/parser/entry.py`，恢复原代码：

```python
workspace_tensor = torch.empty(workspace_size, dtype=torch.uint8, device=device)
```

---

## 步骤 5：结果解读与总结

**⚠️ 重要**：这是最后一步，在完成所有清理和恢复工作后，对检测结果进行解读和总结。

### 5.1 结果解读

**检测结果显示**：

1. **无异常**：提示 "device task 无内存重叠"
2. **存在异常**：提示内存重叠的 device task 以及 leaf function

**错误信息说明**：

| 错误信息 | 含义 | 说明 |
|---------|------|------|
| `memory reuse must happen for full match` | 两个需要内存复用的 rawtensor 范围不一致 | 非内存重叠问题 |
| `memory reuse must happen for same dimension` | 两个内存复用的 rawtensor 的 shape 不一致 | 非内存重叠问题 |

**⚠️ 注意**：上述两种情况非内存重叠，脚本会断言并提示日志信息错误。

### 5.2 输出标准格式

**⚠️ 重要**：必须严格按照以下格式输出检测结果，不要添加推测性分析或建议。

```markdown
## 内存重叠检测结果

### 检测结果
- 是否存在内存重叠：[是/否]
- 重叠的 device task：[task ID/无]
- 重叠类型：[RACE_READ_WRITE / RACE_WRITE_WRITE / RACE_READ_READ / 无]
- 冲突的 leaf task 对：[src_task_id → dst_task_id / 无]

### 修复尝试（如适用）
- 是否尝试扩大 workspace：[是/否]
- 修复结果：[有效/无效]

### 结论
- [✅ 定位到内存重叠问题并已修复 / ✅ 定位到内存重叠问题但修复无效 / ❌ 无内存重叠]
```

**说明**：
- 无内存重叠时，结论使用：❌ 无内存重叠
- 有内存重叠且尝试修复时，根据修复结果选择对应结论

---

## ⭐ 关键注意事项

1. **默认路径**: 步骤 1 使用默认值初始化，无需收集用户输入
2. **路径规范**:
   - 所有路径必须使用绝对路径
   - 步骤 3.1 传递 output **根目录**，不要传递具体的 output_xxx 子目录
3. **修复流程**: 只有检测到内存重叠时，才执行步骤 4 尝试修复
4. **输出规范**: 必须严格按照标准格式输出，不要添加任何推测性分析或建议
5. **超时设置**: bash 命令超时时间设置为 1800000ms（30分钟）
