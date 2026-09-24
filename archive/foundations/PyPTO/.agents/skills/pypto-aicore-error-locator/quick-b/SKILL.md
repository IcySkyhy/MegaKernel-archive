---
name: pypto-aicore-error-locator-quick-b
description: 快捷模式 B — 已知 CCE 文件路径和问题代码行号，直达源码映射。当用户同时提供了 CCE 文件路径和问题行号时触发。不需要 test_cmd。
---

# 快捷模式 B：已知 CCE 文件 + 行号 → 源码映射

用户已通过其他方式定位到 CCE 文件及问题代码行，本模式直达源码映射。

## 前置确认

从用户输入中提取：
- **cce_file**：问题 CCE 文件路径（必须）
- **cce_line**：问题代码行号（必须）
- **run_path**：运行的目录路径（默认当前目录，用于查找 program.json）

> 此模式**不需要 test_cmd**，不需要启用 trace 日志。

## 步骤 1：查找 program.json

从 `run_path/output/` 下查找最新的 program.json（用户之前已运行过测试，program.json 已生成）：

```bash
find <run_path>/output -name "program.json" -path "*/output_*/*" | head -1
```

**⚠️ 重要提示**：
- 若找到 program.json 路径，记录该路径用于步骤 2
- 若未找到，说明之前未运行过测试或 output 目录不存在，**停止执行后续步骤**

## 步骤 2：CCE 行映射到前端源文件

```bash
python3 .agents/skills/pypto-aicore-error-locator/scripts/locate_source_line.py <cce_file> <program_json_path> <cce_line>
```

**参数说明**:
- `<cce_file>`：用户提供的 CCE 文件路径
- `<program_json_path>`：步骤 1 找到的 program.json 路径
- `<cce_line>`：用户提供的问题代码行号

## 输出最终结果

汇总输出以下信息：
- CCE 文件路径
- 问题代码行号
- 问题代码内容
- 前端源代码文件路径（如果映射成功）
- 前端源代码行号（如果映射成功）
- 前端源代码内容（如果映射成功）
