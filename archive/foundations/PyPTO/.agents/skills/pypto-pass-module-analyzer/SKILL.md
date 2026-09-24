---
name: pypto-pass-module-analyzer
description: PyPTO Pass 模块代码分析技能。用于分析 PyPTO pass 的代码，并结合输入文档中的相应内容，生成 Pass 模块分析文档。帮助开发者理解各个模块的接口、功能与特殊场景。当需要理解 PyPTO pass 中某个模块的代码、功能和设计时使用此技能。
---

# PyPTO Pass Module Analyzer Skill

## 概述

本技能用于分析 PyPTO pass 的代码，并结合输入文档中的相应内容，生成 Pass 模块分析文档。

### 使用场景

当需要理解 PyPTO pass 中某个模块的代码、功能和设计时使用此技能。

### 功能概述

- 解析输入文档中的 pass 模块相关资料（如果有的话）
- 在项目中 framework/src/passes 中查找对应源代码
- 结合文档和代码进行分析（如果没有资料输入，直接分析代码）
- 按照 Pass_Analysis_Template.md 模板格式输出结果

### 触发机制

当用户输入包含以下关键字时，自动触发此技能：

- **分析 Pass 代码 XXX**：分析指定 Pass 的代码实现
- **分析 XXX Pass 代码**：分析指定 Pass 的代码实现
- **查看 XXX Pass 代码**：查看指定 Pass 的代码实现
- **分析 XXX Pass 的代码**：分析指定 Pass 的代码实现
- **XXX Pass 代码分析**：分析指定 Pass 的代码实现

**触发示例**：
- "分析 Pass 代码 RemoveRedundantReshape"
- "分析 AutoCast Pass 代码"
- "查看 SubgraphToFunction Pass 代码"
- "分析 RemoveRedundantReshape Pass 的代码"
- "AutoCast Pass 代码分析"

## 输出文件规则

根据需要分析的 pass 在 PVC2_OOO 策略中的位置，输出文件命名规则如下：

**PVC2_OOO 策略中的 pass**：
- 查询路径：`framework/src/passes/pass_mgr/pass_manager.cpp` 中的 `RegDefaultStrategy()` 函数
- 命名格式：`{序号}_{PASS名称}.md`，序号使用两位数字，从 00 开始，具体取值为该 pass 在 PVC2_OOO 策略中的排序
- 例如：00_REMOVE_REDUNDANT_RESHAPE.md, 01_AUTO_CAST.md, ...

**不在 PVC2_OOO 策略中的 pass**：
- 统一以 `99_` 开头
- 命名格式：`99_{PASS名称}.md`
- 例如：99_DYN_ATTR_TO_STATIC.md

### 注意事项

  1. 不要输出大段代码，需要的地方只需要输出函数名
  2. 输出完成后，要重新检查一次输出文档的格式是否统一

## 工作流程

按照输入场景分类，并按照场景下序号依次执行步骤

### 场景 1：指定输入文档时

1. 确认文档描述的 pass 模块
2. 总结文档内容
3. 在项目中 framework/src/passes 中查找对应 pass 模块代码
4. 执行代码分析（详见"代码分析章节"）
5. 执行 Pass 概述分析（详见"Pass 概述分析"）
6. 按照 Pass_Analysis_Template.md 格式生成输出文档
7. 检查最终输出文件的格式，对于格式错误的进行修复

### 场景 2：未指定文档时

1. 询问用户是否要查找全部 pass
2. 查找特定 pass 名称：
    - 搜索 pass_manager.cpp 中 PassName 保存的所有 pass，询问用户想要分析哪个 pass，不要分批展示
    - 根据 pass_manager.cpp 中的注册信息，在 framework/src/passes 中查找用户选择的 pass 的代码
    - 执行代码分析（详见"代码分析章节"）
    - 执行 Pass 概述分析（详见"Pass 概述分析"）
    - 按照 Pass_Analysis_Template.md 格式生成输出文档
    - 检查最终输出文件的格式，对于格式错误的进行修复
3. 查找全部：
    - 搜索 pass_manager.cpp 中 PassName 保存的所有 pass
    - 根据 pass_manager.cpp 中的注册信息，依次遍历每个 pass 的代码
    - 对每个 pass 执行代码分析（详见"代码分析章节"）
    - 执行 Pass 概述分析（详见"Pass 概述分析"）
    - 按照 Pass_Analysis_Template.md 格式生成输出文档
    - 检查最终输出文件的格式，对于格式错误的进行修复

## 代码分析章节

### 1. 基础代码分析

在分析 Pass 代码时，需要重点关注以下内容：

- **入口函数识别**：找出 Pass 的入口处理函数RunOnFunction并进行分析描述
- **Checker识别分析**：识别模块的Prechecker/Postchecker并进行分析描述
- **关键函数梳理**：理解代码的关键业务函数，对于每个关键函数进行如下的分析：
  -- **输入输出分析**：分析关键业务函数输入输出信息
  -- **辅助函数分析**：分析被核心函数调用的辅助函数及其作用
  -- **算法复杂度分析**：分析关键业务函数的时间复杂度和空间复杂度（需要注意算法复杂度与function中的Operation、Tensor或Incast/Outcast的关系）
- **关键业务梳理**：理解模块的主要处理场景，对于每个场景进行如下的分析：
  -- **逻辑视图分析**：通过Operation与Tensor组合成逻辑视图，展示该pass对此场景的修改
  -- **场景描述**: 通过简单的语言描述这个业务场景

### 2. OPCode 特判分析

在分析代码时，需要特别搜索和分析以下内容：

#### 1. 重点关注的视图类 OPCode

以下三个视图类 OPCode 在 Pass 中经常需要特殊处理，**必须**在分析时重点记录它们的特判场景：

- **OP_VIEW**：视图操作，用于创建视图
- **OP_ASSEMBLE**：组装操作，用于组装张量
- **OP_RESHAPE**：重塑操作，用于改变张量形状

#### 2. 其他 OPCode 特判分析

**搜索目标**：
- 搜索代码中所有 `GetOpcode()`、`GetOpCode()` 相关的判断
- 搜索 `Opcode::OP_` 相关的常量使用
- 搜索 `if (op->GetOpcode() == Opcode::OP_XXX)` 或类似的特判逻辑

**分析要点**：
- 记录每个特判的场景和条件
- **详细记录视图类 OPCode 的特殊处理逻辑**
- 将这些特判场景整理到输出文档的"注意事项"模块中

#### 3. 特判类型识别

在分析代码时，识别以下类型的 OPCode 特判：

- **直接特判**：`if (op->GetOpcode() == Opcode::OP_XXX)`
- **集合特判**：`if (xxxOps.count(op->GetOpcode()) > 0)`
- **否定特判**：`if (op->GetOpcode() != Opcode::OP_XXX)`
- **多重特判**：`if (op->GetOpcode() == Opcode::OP_XXX || op->GetOpcode() == Opcode::OP_YYY)`

#### 4. 特判场景记录

对于每个 OPCode 特判，记录以下信息：

- **特判的 OPCode**：具体是哪个或哪些操作码
- **特判条件**：判断的具体条件
- **特判位置**：在哪个函数或代码块中
- **处理逻辑**：特判后执行的操作
- **业务含义**：这个特判的业务目的

**特别强调**：
- 如果特判涉及 **OP_VIEW**、**OP_ASSEMBLE**、**OP_RESHAPE**，必须详细记录
- 标注这些视图类 OPCode 的特殊处理逻辑
- 说明为什么这些视图类 OPCode 需要特殊处理

#### 5. 特判分类

将特判按照处理方式进行分类：

- **跳过处理**：某些 OPCode 被跳过不处理
- **特殊处理**：某些 OPCode 需要特殊逻辑处理
- **禁止操作**：某些 OPCode 不允许出现
- **兼容处理**：某些 OPCode 需要兼容性处理

## Pass 概述分析

按照 **Pass 名称**、**Pass 类型**、**简要描述**这几个维度进行分析（输出结果时对应到 Pass_Analysis_Template.md 中的“Pass 概述”章节）

## 输出格式

按照 Pass_Analysis_Template.md 模板格式输出，包含以下章节：

1. Pass 概述
2. 代码分析
   - 主要功能
   - 处理流程
     - PreCheck 阶段
     - RunOnFunction 阶段
     - PostCheck 阶段
     - 关键函数的核心逻辑
     - Pass 业务核心逻辑图
3. 业务分析
   - 适用场景
   - 优化效果
   - 典型应用场景
4. OPCode 特判分析
5. 总结
6. 相关文件
7. 附录
