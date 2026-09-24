---
name: precision-occasional
description: 偶现精度问题排查技能。当精度问题无法稳定复现（偶现、随机失败、时好时坏）时使用，通过独立开关逐项关闭框架功能，观察精度是否恢复，定位偶现问题的归属组件。触发词：偶现、不稳定、无法复现、随机失败、时好时坏、occasional。
---

# 偶现精度排查

> **独立场景，不在全自动排查流程中**。仅当精度问题无法稳定复现时触发。

通过独立开关逐项关闭框架功能，每次只改一个变量，观察精度问题是否恢复（不再复现）。精度恢复 → 该功能为问题归属。

> **编译说明**：修改框架 C++ 代码（开关 1、2）需重新编译安装；修改 `runtime_options`（开关 3）无需重新编译，直接执行即可。

## 流程

```
偶现精度问题触发
       │
       ▼
开关 1: 开启同步调试
       │  重新编译安装 → 执行算子 10 次
       ├── 精度通过（10 次全部通过，不再复现）→ 同步问题 ──→ 结束
       │
       │ 仍有精度失败
       ▼
开关 2: 关闭 GM 内存复用
       │  重新编译安装 → 执行算子 10 次
       ├── 精度通过（10 次全部通过，不再复现）→ GM 内存复用问题 ──→ 结束
       │
       │ 仍有精度失败
       ▼
开关 3: 关闭 stitch 融合
       │  直接执行算子（无需重新编译）→ 执行算子 10 次
       ├── 精度通过（10 次全部通过，不再复现）→ stitch 融合问题 ──→ 结束
       │
       │ 仍有精度失败 → 记录全部开关的测试现象上报，可能需要框架侧深度排查
       └── 结束
```

## 开关详情

### 开关 1：开启同步调试

**修改文件**：`framework/src/passes/block_graph_pass/insert_sync.h`

**修改内容**：将 `bool enableDebug_{false}` 改为 `bool enableDebug_{true}`

**验证目标**：排除同步缺失导致的偶现数据竞争。`enableDebug_` 设为 true 后，框架在同步点插入额外的同步指令，确保所有计算完成后再继续。

**操作步骤**：

1. 修改代码：
   ```cpp
   // insert_sync.h
   bool enableDebug_{true};    // 原为 false
   ```

2. 重新编译安装：
   ```bash
   python3 build_ci.py --clean --no_isolation && bash build_out/cann-pypto_*.run --full -q --pylocal
   ```

3. 执行算子 10 次，统计执行结果（通过次数 / 失败次数）

4. 判定：
   - 10 次全部通过（不再复现）→ **同步问题** → 结束
   - 仍有精度失败 → 恢复原代码，进入开关 2

### 开关 2：关闭 GM 内存复用

**修改文件**：`framework/src/passes/block_graph_pass/memory_reuse/global_memory_reuse.cpp`

**修改内容**：在 `Allocator::Init()` 中设置 `skipReuseJudgment_ = true`

**验证目标**：排除 GM 内存复用导致的偶现数据覆盖。`skipReuseJudgment_` 设为 true 后，跳过 GM 内存复用判断，每张 tensor 独占 GM 内存，不复用其他 tensor 释放的内存。

**操作步骤**：

1. 修改代码：
   ```cpp
   // global_memory_reuse.cpp — Allocator::Init()
   skipReuseJudgment_ = true;
   ```

2. 重新编译安装：
   ```bash
   python3 build_ci.py --clean --no_isolation && bash build_out/cann-pypto_*.run --full -q --pylocal
   ```

3. 执行算子 10 次，统计执行结果（通过次数 / 失败次数）

4. 判定：
   - 10 次全部通过（不再复现）→ **GM 内存复用问题** → 结束
   - 仍有精度失败 → 恢复原代码，进入开关 3

### 开关 3：关闭 stitch 融合

**修改文件**：算子实现文件的 `@pypto.frontend.jit` 装饰器

**修改内容**：在 `runtime_options` 中设置 `"stitch_function_max_num": 1`

**验证目标**：排除 stitch 融合调度导致的偶现精度异常。`stitch_function_max_num` 控制可拼接的最大子函数数，设为 1 后每个 task 只能拼接 1 个子函数，实质关闭多函数融合调度。

**操作步骤**：

1. 修改代码：
   ```python
   @pypto.frontend.jit(
       runtime_options={
           "run_mode": pypto.RunMode.NPU,
           "stitch_function_max_num": 1,    # 关闭 stitch 融合
       }
   )
   def your_kernel(...):
   ```

2. 直接执行算子（**无需重新编译**，`runtime_options` 在运行时读取）

3. 执行算子 10 次，统计执行结果（通过次数 / 失败次数）

4. 判定：
   - 10 次全部通过（不再复现）→ **stitch 融合问题** → 结束
   - 仍有精度失败 → 恢复原配置，记录全部开关的运行次数和复现情况并上报，可能需要框架侧深度排查

## 关键原则

1. **每次只改一个变量**：不同时修改多个开关，避免结论混淆
2. **编译区分**：框架 C++ 代码修改（开关 1、2）需重新编译安装；`runtime_options` 修改（开关 3）无需重新编译
3. **每个开关运行 10 次**：偶现问题需要足够样本统计复现概率，统计通过次数与失败次数
4. **测试完成后恢复所有代码改动**
5. **报告记录**：每个开关的修改内容、运行次数（10 次）、通过次数、失败次数、复现概率
