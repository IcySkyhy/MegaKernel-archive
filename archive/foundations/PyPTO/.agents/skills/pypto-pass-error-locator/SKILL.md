---
name: pypto-pass-error-locator
description: PyPTO Pass 模块错误诊断技能。包含错误定位、原因分析和提供问题修复建议，提供从问题定位到修复建议的完整工作流程。当遇到 PyPTO Pass 模块抛出错误时使用此技能。触发词：定位 pass 错误、pass 模块异常、pass 报错、pass 失败、pass 异常。
license: 完整条款见 LICENSE.txt
---

# PyPTO Pass 错误诊断与修复技能

本技能提供 PyPTO Pass 模块错误的完整诊断和修复建议能力，从错误原因定位到提供修复建议的端到端解决方案。

## 使用场景

当某业务场景执行失败，需要分析定位某pass抛出异常原因并获取修复建议时使用此技能。

## 功能概述

- 解析 PyPTO 运行日志，提取错误信息
- 识别错误相关的 Pass 模块
- 分析错误产生的根本原因
- 定位问题代码位置
- 提供修复建议

## 前置条件

使用此技能前需要满足以下条件：

1. **环境要求**
   - PyPTO 开发环境已正确配置
   - 可访问 PyPTO 源代码目录
   - 具有日志文件读取权限

2. **输入要求**
   - 用户必须提供复现问题的执行命令
   - 可选：提供相关的测试用例或脚本路径

3. **依赖技能**
   - `pypto-environment-setup`：用于检查环境状态
   - `pypto-pass-module-analyzer`：用于分析对应 Pass 模块实现
   - `pypto-pass-workflow-analyzer`：用于分析 Pass 业务流和上下游依赖

## 触发机制

当用户输入包含以下错误日志或关键字时，自动触发此技能：

- **定位 pass 错误**：定位 Pass 抛出异常的具体原因，并提供修复方案
- **pass 模块异常**：定位 Pass 抛出异常的具体原因，并提供修复方案
- **pass 报错**：定位 Pass 抛出异常的具体原因，并提供修复方案
- **pass 失败**：定位 Pass 抛出异常的具体原因，并提供修复方案
- **pass 异常**：定位 Pass 抛出异常的具体原因，并提供修复方案

**触发示例**
- 执行 python3 build_ci.py -c -f=cpp -u=NBufferMergeTest.TestMode4 异常，pass 报错
- 执行 python3 test.py 异常，分析 pass 失败原因

## 工作流程

### 步骤 1：问题复现

1. 如果用户执行的是python相关的脚本，开启图编译阶段调试模式开关（开启调试模式编译时会默认在`$(pwd)/output`目录下输出各 pass 模块的`计算图`和`IR`等文件）：

   ```python
    @pypto.frontend.jit(
        debug_options={"compile_debug_mode": 1, "runtime_debug_mode": 1}
    )
    ```

2. 设置日志相关环境变量：
   - 设置日志输出目录环境变量：`export ASCEND_PROCESS_LOG_PATH=$(pwd)/logs/$(date +%Y%m%d%H%M%S) `
   - 设置日志输出级别环境变量：`export ASCEND_GLOBAL_LOG_LEVEL=0 `

3. 根据用户提供的执行命令，复现用户问题

**重要**: 上述步骤中的 `设置日志相关环境变量` 与 `根据用户提供的执行命令，复现用户问题`，必须同一个会话中完成，否则环境变量的设置无法生效。

**验证检查点**：
- [ ] 问题成功复现
- [ ] 获得可复现的执行命令
- [ ] 记录复现时的环境信息
- [ ] 记录用户代码文件路径

### 步骤 2：关键信息获取

#### 2.1 获取日志内容

**日志查找策略：**

1. 从 `$ASCEND_PROCESS_LOG_PATH/debug/plog` 目录下获取 `pypto-*.log`
2. 优先查找pass模块相关的[ERROR]、[WARN]级别的日志

**日志文件验证：**
- 确认日志文件包含 [PASS] 标记
- 确认日志文件包含文件路径和行号信息

**pass 中断但无异常日志的特殊处理：**
1. 用户执行的是python相关的脚本时，参考 `计算图和IR查找策略` 查找计算图和IR信息，若没有 `CodegenPreproc` 模块的相关输出，即可判定 pass 未执行到最后一个模块。
2. 若日志中没有 `[ERROR]` 日志，且pass执行中断，在最后中断的pass模块中增加异常捕获代码，对该 pass 的 `RunOnFunction` 增加临时异常捕获代码，至少覆盖：
   - `catch (const std::exception& e)`，打印 `e.what()`
   - `catch (...)`，打印 unknown exception
   - 如有必要，可在 pass 内关键步骤前后增加临时日志，缩小中断位置
3. 加入临时代码后，重新编译并安装 pypto 包：`python3 build_ci.py --clean --no_isolation && bash build_out/cann-pypto_*.run --full -q --pylocal
4. 在同样的日志环境变量配置下重新执行用户复现脚本。
5. 上述异常捕获和临时日志仅用于定位；定位完成后，将临时诊断代码回退。

**验证检查点**：
- [ ] 成功定位日志文件
- [ ] 日志内容可读取
- [ ] 日志内容包含错误或警告信息
- [ ] 日志内容包含 Pass 模块信息
- [ ] 已检查默认策略最后一个 pass `CodegenPreproc` 是否执行完成
- [ ] 已在最后中断 pass 的 `RunOnFunction` 中补充临时异常捕获或关键日志
- [ ] 已重新编译安装并复现，确认是否拿到新的异常日志

#### 2.2 获取计算图和IR（用户执行python脚本场景）

**计算图和IR查找策略：**

1. 开启 `compile_debug_mode` 后，计算图和 IR 默认输出到 `$(pwd)/output` 目录下。
2. 每次执行会在 `output` 目录下生成一个新的子目录，目录名通常类似：`output_20260417_153300_744883_2765832_C0A8451A`。
3. 进入本次执行对应的子目录后，优先查找 `Pass_*` 目录；每个 `Pass_xxx` 目录下存放对应 pass 模块执行前后的计算图 JSON 和 IR 文件。
4. 需要至少收集：
   - 当前报错 pass 对应目录下的 `Before` 计算图和 IR
   - 当前报错 pass 对应目录下的 `After` 计算图和 IR（若已生成）
   - 上游相邻 pass 的输出文件，用于回溯异常首次出现的位置

**执行异常时的特殊处理：**

1. 如果某个 pass 模块执行异常并在该 pass 内中断，必须先检查该 `Pass_xxx` 目录下的 `After` 计算图是否成功打印。
2. 如果 `After` 计算图未成功打印，不得直接基于 `Before` 图下结论；必须参考 `references/pass-error-analysis-guide.md` 中的“异常前补打计算图”相关流程，定位最近异常点并插入临时 `DumpJsonFile` 代码。
3. 插入临时 `DumpJsonFile` 代码后，必须重新编译并安装 pypto 包，再重新执行用户复现脚本。
4. 常规安装命令：`python3 build_ci.py --clean --no_isolation && bash build_out/cann-pypto_*.run --full -q --pylocal`
5. 重新安装完成后，重新执行用户复现脚本，确认临时补打的计算图已成功生成，再继续后续异常分析。

**验证检查点：**
- [ ] 成功定位本次执行对应的 `output_*` 子目录
- [ ] 成功定位当前报错 pass 对应的 `Pass_*` 目录
- [ ] 成功获取当前 pass 的 `Before` 计算图和 IR
- [ ] 已确认当前 pass 的 `After` 计算图是否存在
- [ ] 若 `After` 计算图缺失，已按参考流程插入临时 `DumpJsonFile` 代码
- [ ] 插桩后已按常规安装方式重新编译安装 pypto 包
- [ ] 已重新执行用户复现脚本并拿到补打后的计算图

### 步骤 3：解析日志关键信息

**日志格式示例**：

以如下 pass 模块打印的日志格式为例，分段解析：
```
[ERROR] PYPTO(638465):2026-03-16 10:02:24.711 [n_buffer_merge.cpp:530][PASS]:[NBufferMerge.Config]:The VEC_NBUFFER_SETTING key -3 is incorrect; Please set keys of VEC_NBUFFER_SETTING between -1 and max hashOrder 0.
```

**解析字段**：

1. 日志级别：[ERROR]
2. 模块名及进程：PYPTO(638465)
3. 日志打印时间：2026-03-16 10:02:24.711
4. 代码文件及行号：[n_buffer_merge.cpp:530] 此处 `n_buffer_merge.cpp` 为文件名，`530` 为打印该日志代码所在行号
5. 所属模块类型：[PASS]
6. 具体模块信息：[NBufferMerge.Config]，此处 `NBufferMerge` 可能是pass模块名，也可能是其他工具模块名，如果不是pass模块名称要根据日志上下文分析是哪个pass模块调用的该方法
7. 日志内容：The VEC_NBUFFER_SETTING key -3 is incorrect; Please set keys of VEC_NBUFFER_SETTING between -1 and max hashOrder 0.

**多行错误处理**：
- 检查错误信息是否跨越多行
- 合并相关联的错误上下文
- 提取完整的错误堆栈信息

**日志级别处理**：
- `[ERROR]`：必须处理的关键错误
- `[WARN]`：可能导致问题的警告信息
- `[INFO]`：辅助调试的信息日志

**验证检查点**：
- [ ] 日志级别正确识别
- [ ] 代码位置准确提取
- [ ] Pass 模块名称正确解析
- [ ] 错误内容完整提取
- [ ] 时间戳格式正确解析

### 步骤 4：异常分类

#### 4.1 常见异常类型

| 异常类型 | 日志特征 | 关键信息提取 |
|---------|---------|-------------|
| **参数配置异常** | `vec_nbuffer_setting`, `cube_l1_reuse_setting`, `cube_nbuffer_setting`, `sg_set_scope` | 异常参数名、参数配置值 |
| **算子约束违反** | `constraint violated`, `invalid parameter` | opcode类型、op_magic信息 |
| **段错误** | `segment fault` | 调用堆栈信息 |
| **属性缺失** | `attribute not found`, `missing attribute`, `required attribute`, `get attribute failed` | 属性名称、算子名称、期望属性类型、缺失的属性列表 |
| **计算图成环** | `cycle detected`, `graph cycle`, `topological sort failed`, `circular dependency` | 成环节点、循环路径、依赖关系、拓扑排序失败节点 |
| **索引越界** | `index out of range`, `invalid index` | 索引值、范围、张量形状 |


#### 4.2 将异常进行分类
- 分析用户提供的测试代码，了解用户业务场景
- 根据日志中获取的异常代码位置，执行以下可执行动作：
  - 固定命令抓取上下文：`sed -n '<line-20>,<line+10>p' <file>`
  - 输出三项证据：触发条件、关键分支变量、返回码
- 使用 `pypto-pass-module-analyzer` 技能分析理解对应pass模块整体业务逻辑
- 根据日志内容结合代码逻辑，推断异常类型，异常类型及特征如常见异常类型表格中所示

**验证检查点**：
- [ ] 异常类型正确分类
- [ ] 业务场景理解准确
- [ ] 代码上下文分析完整

### 步骤 5：异常分析

#### 5.1 按类型进行异常分析（必须执行）
- 读取 `references/pass-error-analysis-guide.md` 文件，其中已包含日志、计算图 JSON 的统一分析流程
- 根据异常类型定位到对应的分析指导章节
- 严格按照章节定义的步骤顺序执行，记录每步结果
- 不得跳过或合并步骤

#### 5.2 汇总分析结果给出修复建议
- 生成根因链：现象(日志) -> 触发位置(源码切片) -> 状态异常(计算图/IR变化) -> 违反规则(约束文档)
- 若初始日志无异常但通过临时异常捕获补充得到了新错误信息，根因链必须明确区分：
  - 首次外显现象：pass 中断且 `After` 图缺失
  - 补充诊断手段：在目标 pass 的 `RunOnFunction` 增加异常捕获/临时日志后重新复现
  - 最终异常证据：补充捕获后新增的错误码、错误文本、源码行号
- 汇总异常定位过程中的关键信息及分析结果，形成分析报告，格式参考 [报告模板](references/pass-error-report-template.md)
- 给出可行的修复建议（如：增加判空、修改算子映射逻辑、调整内存分配策略）

**验证检查点**：
- [ ] 分析指导获取准确
- [ ] 错误原因分析准确
- [ ] 代码位置精确定位
- [ ] 修复建议合理可行

## 输出格式

报告格式详见 [PyPTO Pass 错误分析报告模板](references/pass-error-report-template.md)。

## 输出成功标准

一个有效的错误分析报告必须满足：
- 报告包含基本信息、错误位置、错误信息、原因分析、修复建议
- 代码位置精确到文件和行号
- 修复建议具体可执行
- 错误原因分析基于日志和源码证据
- 根因链完整（现象 -> 触发位置 -> 状态异常 -> 违反规则）

## 输出校验与回退

### 校验方式

输出报告后，执行以下校验：

1. **格式校验**：确认报告包含六个必需章节（基本信息、错误位置、错误信息、原因分析、修复建议、附录）
2. **证据链校验**：确认根因链完整且各环节有对应证据（日志片段、源码行号、计算图差异、文档约束）
3. **可执行性校验**：确认修复建议包含具体操作步骤而非泛化描述

### 失败回退

当校验失败时，按以下路径回退：

| 失败类型 | 回退步骤 | 回退原因 |
|---------|---------|---------|
| 格式不完整 | 回退到步骤 5（异常分析） | 重新汇总分析结果 |
| 证据链断裂 | 回退到步骤 2（关键信息获取） | 补充缺失的日志或计算图证据 |
| 修复建议不可执行 | 回退到步骤 4（异常分类） | 重新定位根因并细化修复方案 |

回退后必须记录：回退原因、缺失内容、补充动作。

## 性能优化建议

处理大规模日志文件时：
1. 使用流式读取，避免一次性加载整个文件
2. 优先搜索 ERROR 级别日志，减少处理范围
3. 使用多线程并行处理多个日志文件
4. 缓存已解析的日志信息，避免重复解析

## 参考文档

### 核心文档
- [PyPTO Pass 异常分析流程指导](references/pass-error-analysis-guide.md)
- [PyPTO IR分析指导](references/ir-analysis-guide.md)
- [PyPTO Pass 错误分析报告模板](references/pass-error-report-template.md)
- [查看计算图](../../../docs/zh/tools/computation_graph/查看计算图.md)

### 相关技能
- [pypto-environment-setup](../pypto-environment-setup/SKILL.md)
- [pypto-pass-module-analyzer](../pypto-pass-module-analyzer/SKILL.md)
- [pypto-pass-workflow-analyzer](../pypto-pass-workflow-analyzer/SKILL.md)

### API文档
- [Pass配置API](../../../docs/zh/api/config/pypto-set_pass_options.md)
