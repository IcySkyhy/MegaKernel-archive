---
name: pypto-machine-workspace
description: PyPTO Workspace 内存异常偏大问题自动诊断技能。通过系统化的日志分析流程，逐层拆解内存预算结构，定位导致 workspace 内存超大的根因组件和具体 Tensor，并给出配置调优或问题归属建议。当用户报告 workspace 内存异常、NPU OOM、rtMalloc failed、内存申请失败、workspace 过大等问题时使用此技能。
---
# Workspace 内存异常偏大自动诊断

此技能用于诊断 PyPTO 运行时 workspace 内存异常偏大的问题。通过逐层拆解内存预算结构、分析日志中各子项占比，定位导致内存超大的具体组件和 Tensor，并给出配置调优或问题归属建议。

## 使用场景

当运行 PyPTO 算子或模型时遇到以下报错之一，即可使用此技能：

- **torch 申请失败**：`torch.OutOfMemoryError: NPU out of memory. Tried to allocate X GiB ...`
- **device 内存申请失败**：`rtMalloc failed. size:XXXXXXXXXX`
- **运行时分配失败**：`Memory not enough(alloc ...)` 或 `SeqWsAllocator cannot allocate memory`
- **用户主动反馈** workspace 占用异常偏大

## 前置条件

1. **环境要求**
   - PyPTO 开发环境已正确配置，可编译 whl 包
   - 可访问 PyPTO 源代码目录
   - 具有日志文件读取权限

2. **输入要求**
   - 用户必须提供复现问题的执行命令
   - 用户需提供 PyPTO 项目根目录路径

3. **依赖技能**
   - `pypto-environment-setup`：用于检查环境状态

## 触发机制

当用户输入包含以下关键字或相关内容时，自动触发此技能：

- **workspace 内存异常/过大**：诊断 workspace 内存预算哪一项导致了总量异常
- **NPU out of memory / rtMalloc failed**：定位内存分配失败的根因
- **显存不足 / 内存分配失败**：分析 workspace 各子项大小并给出建议
- **workspace 占用偏大如何优化**：分析日志并提供配置调优方案

**触发示例**：
- 运行用例报 NPU out of memory，workspace 太大了
- rtMalloc failed，帮我定位 workspace 内存为什么这么大
- workspace 占用超过 5GB，帮我分析原因

---

## 背景知识

> ⚠️ **重要**：此部分为诊断必备的背景知识，agent 在执行诊断前必须理解这些概念。

### 内存分配阶段

Workspace 内存分配由以下阶段组成：

1. **图编译阶段**：在 `dev_encode.cpp` 的 `EncodeDevAscendProgram` 中估算各类内存池预算
2. **Launch 阶段（用户指定）**：通过 `dynWorkspaceSize` 指定额外动态内存（一般不会是此处问题）
3. **Launch 阶段（实际分配）**：在 `device_launcher.h` 中通过 `devMem.AllocDev(devProg->workspaceSize, ...)` 一次性分配
4. **Device 运行时**：在 `dev_workspace.h` 中按预算对内存池进行 suballocation

实际发生 malloc 的只有阶段 3，内存池超大通常说明阶段 1 的预算估算出了问题。

### Workspace 内存预算结构

定义于 `framework/src/machine/utils/dynamic/dev_encode_program.h` 的 `memBudget` 结构：

```
workspaceSize = memBudget.Total()
             = tensor.Total() + aicoreSpilled.Total() + debug.dumpTensor + debug.leafDump + metadata.Total()

tensor.Total() = rootInnerSpilledMem                    -- Root Function Inner Tensor 内存
               + devTaskInnerExclusiveOutcasts           -- DeviceTask 内部 Exclusive Outcast 内存
               + MaxOutcastMem() * (devTaskBoundaryOutcastNum + devTaskInnerTemporalOutcastNum)

MaxOutcastMem() = max(maxStaticOutcastMem, maxDynamicAssembleOutcastMem)

metadata.Total() = general + stitchPool
```

### 关键日志标签

诊断日志以 `[workspaceSize]` 为统一前缀（需使用包含该日志的 whl 包版本），主要包括：

| 日志内容 | 说明 |
|---------|------|
| `Metadata=..., workspaceSize=..., tensor=..., aicoreSpillen=...` | 总量与各大类拆分 |
| `Tensor:rootInnerSpilledMem=A, devTaskInnerOutCasts=B, slotted=CxD(slots)` | A=rootInnerSpilledMem, B=devTaskInnerExclusiveOutcasts, C=MaxOutcastMem(), D=boundary+innerTemporal slot 数 |
| `OutcastSlots: boundary=X, innerTemporal=Y` | Boundary / Inner Temporal slot 数分项 |
| `MaxRootInnerMem is ..., maxDevTaskInnerExclusiveOutcastMem is ...` | 滚动取 max 的中间值（经过 unroll/stitch 膨胀后） |
| `Rootfunction: <name> ->MaxRootInnerMem is ..., maxDevTaskInnerExclusiveOutcastMem is ...` | 每个 Root Function 级别的内存贡献 |
| `Root=[<name>], symbol=[<sym>],rawmagic=[<id>]: staticMemReq=[<size>] is too larger` | shape 超过 512×512 的超大 Tensor 告警 |
| `ItemPoolMemSize is: ..., vectorMemSize is: ..., slotAllocatorMemSize is ...` | 元数据内存子项明细 |
| `workspace of generalMetadataSlotSize is ...` | general 元数据槽位总大小 |
| `RequiredSlabNum[N] is ...` | 各类型 Slab 所需数量 |

---

## 工作流程

**⚠️ 重要提示**：将 bash 运行命令超时时间设置为 1800000ms

### 步骤 1：收集必要信息（必须执行）

**⚠️ 重要：第一步必须使用 `question` 工具向用户收集信息，严禁猜测或使用默认值。**

使用 `question` 工具收集以下信息：

- **pypto_path**：PyPTO 项目的根目录路径（绝对路径）
- **test_cmd**：触发内存异常的完整测试命令
- **run_path**：运行测试命令的目录路径（绝对路径）
- **error_msg**：用户看到的报错信息（torch OOM / rtMalloc failed / 其他）
- **has_diag_logs**：用户是否已有包含 `[workspaceSize]` 日志的运行结果（是/否）

将收集的路径全部转换成绝对路径。

**判断分支**：
- 若用户已有诊断日志（`has_diag_logs=是`）→ 跳至步骤 3
- 否则 → 继续步骤 2

---

### 步骤 2：开启诊断日志并复现问题

需要修改三处配置，重新编译 whl 包后复现问题。

#### 2.1 设置 INFO 日志级别

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1
export GLOBAL_LOG_LEVEL=1
export ASCEND_PROCESS_LOG_PATH=<run_path>/ws_diag_logs
```

#### 2.2 打开计算图 dump

读取 `<pypto_path>/framework/src/interface/configs/tile_fwk_config.json`，将 `dump_graph` 设置为 `true`：

```json
"dump_graph": true
```

#### 2.3 编译安装并复现

```bash
cd <pypto_path> && python3 build_ci.py -f python3 --disable_auto_execute
pip install build_out/pypto*.whl --force --no-deps
cd <run_path> && <test_cmd>
```

**验证检查点**：
- [ ] 问题成功复现（出现 OOM 或 rtMalloc 报错）
- [ ] `<run_path>/ws_diag_logs/debug/` 目录下生成了日志文件
- [ ] `<run_path>./output/pass/` 目录下生成了计算图文件

---

### 步骤 3：提取 workspace 总体大小构成

在日志目录 `<log_path>/debug/` 下搜索 `[workspaceSize]` 关键字，提取以下两行核心日志：

```bash
grep -r "\[workspaceSize\]" <log_path>/debug/ | grep -E "workspaceSize=|Tensor:rootInnerSpilledMem="
```

**期望获取到的日志格式**：

```
[workspaceSize] Metadata=<M>, workspaceSize=<TOTAL>, tensor=<T>, aicoreSpillen=<S>, debug.DumpTensor=<DD>, leafDumpWorkspace=<LD>.
[workspaceSize] Tensor:rootInnerSpilledMem=<A>, devTaskInnerOutCasts=<B>, slotted=<C>x<D>(slots).
[workspaceSize] OutcastSlots: boundary=<X>, innerTemporal=<Y>.
```

**解析日志值**：

| 变量 | 含义 | 来源 |
|------|------|------|
| TOTAL | workspace 总大小 | `memBudget.Total()` |
| T | Tensor workspace 总大小 | `memBudget.tensor.Total()` |
| M | 元数据总大小 | `memBudget.metadata.Total()` |
| S | AICore 栈溢出内存 | `memBudget.aicoreSpilled.Total()` |
| A | rootInnerSpilledMem | `memBudget.tensor.rootInnerSpilledMem` |
| B | devTaskInnerExclusiveOutcasts | `memBudget.tensor.devTaskInnerExclusiveOutcasts` |
| C | MaxOutcastMem() | 单个 slotted Outcast 最大大小 |
| D | devTaskBoundaryOutcastNum + devTaskInnerTemporalOutcastNum | slotted 总 slot 数 |
| X/Y | boundary / innerTemporal slot 数 | 见 `OutcastSlots:` 日志 |

**⚠️ 重要**：如果搜索不到 `[workspaceSize]` 前缀的日志，说明 whl 包版本不包含此日志。此时改为搜索以下已有日志（位于 `device_launcher.h`）：

```bash
grep -r "workspaceSize=" <log_path>/debug/ | grep "tensor="
grep -r "Tensor:rootInnerSpilledMem=" <log_path>/debug/
```

---

### 步骤 4：判断异常大类

根据步骤 3 提取的值，按以下决策树判断问题所在大类：

```
workspaceSize = TOTAL
├── tensor (T) 占比 > 80%？
│   ├── 是 → 进入步骤 5（Tensor Workspace 分析）
│   └── 否 ↓
├── metadata (M) 占比 > 30%？
│   ├── 是 → 元数据内存问题，建议联系 machine 同事
│   │         （搜索日志中 "Memory not enough" + "WsProperty:metadata" 或 "Slab alloc null"）
│   └── 否 ↓
├── aicoreSpillen (S) 占比 > 30%？
│   ├── 是 → AICore 栈溢出问题，需检查算子的 stackWorkSpaceSize
│   └── 否 ↓
├── debug.DumpTensor 或 leafDumpWorkspace 非零？
│   ├── 是 → 调试模式开销，确认是否关闭了调试选项
│   └── 否 → 综合分析：按各组件 workspace 占比降序排列，优先处理占比最高的组件（阈值：单项 > 30% 或多项合计 > 60%）
```

**⚠️ 重要**：绝大多数 workspace 偏大问题集中在 **Tensor Workspace**，直接进入步骤 5 是最常见路径。

---

### 步骤 5：分析 Tensor Workspace 各项占比

从步骤 3 的 Tensor 日志中提取 A、B、C、D 值，计算各项占比：

```
Tensor 总量 T = A + B + C × D

rootInnerSpilledMem 占比      = A / T
devTaskInnerOutCasts 占比 = B / T
BoundaryOutcast 占比 = (C × D) / T
```

根据占比判断进入不同分支：

#### 情况一：rootInnerSpilledMem (A) 或 devTaskInnerOutCasts (B) 为主要贡献者

提取 Root Function 级别日志：

```bash
grep -r "\[workspaceSize\].*Rootfunction:" <log_path>/debug/
```

或搜索不带前缀的日志：

```bash
grep -r "maxRootInnerMem is\|MaxRootInnerMem is" <log_path>/debug/
```

**分析要点**：
1. 对比各 Root Function 的 `MaxRootInnerMem` 和 `maxDevTaskInnerExclusiveOutcastMem`，找到贡献最大的 Root Function
2. 检查该 Root Function 名称中是否包含 `Unroll` 标记（如 `_Unroll8`），判断 unroll 次数
3. 内存膨胀关系（近似公式，encode 与 runtime 均按 root function 个数计量 k_eff）：
   ```
   rootInnerSpilledMem ≈ max_root(AlignUp(rootInnerTensorWsMemoryRequirement) × k_eff)
   devTaskInnerOutCasts ≈ max_root(AlignUp(exclusiveOutcastWsMemoryRequirement) × k_eff)
   k_eff = stitch_function_max_num（默认）或由 max_workspace_kb 反推后 cap 至 stitch_function_max_num
   devTaskBoundaryOutcastNum = E×2 + A×2（boundary 池）
   devTaskInnerTemporalOutcastNum = A × k_eff（innerTemporal 池）
   runtimeOutcastCacheDepth = min(k_eff, MAX_STITCH_FUNC_NUM)
   runtimeOutcastPoolSize = totalSlot × (runtimeOutcastCacheDepth + 1) × parallelism
   ctrlflow backup 容量 ≥ devTaskBoundaryOutcastNum + devTaskInnerTemporalOutcastNum
   ```
4. 若多个 Root Function 的值均匀偏大 → 可能是 `stitch_function_max_num` 或 `max_workspace_kb` 配置过大
5. 若单个 Root Function 的值远超其他 → 该 Root Function 内存需求本身偏高，需分析 pass 内存复用

**配置调优建议**（作为临时缓解手段）：

| 配置项 | 影响范围 | 调优方向 |
|--------|----------|----------|
| `max_workspace_kb` | k_eff、innerTemporal slot 数、runtimeOutcastCacheDepth、runtime submit 深度（内存驱动） | **推荐优先配置**；按 encode 日志推荐值设置，使能内存驱动模式 |
| `stitch_function_max_num` | rootInnerSpilledMem、devTaskInnerOutCasts、innerTemporal slot 数、runtime submit 深度（非内存驱动） | 降低此值以减少并行度换取内存 |
| `unroll_list` / max_unroll | rootInnerSpilledMem、devTaskInnerOutCasts | 减小 unroll 数；使用 unroll_list 时建议配合 `max_workspace_kb` |

#### 情况二：Boundary Outcast 内存 (C × D) 为主要贡献者

C（`MaxOutcastMem()`）异常大是最常见的原因。由于 Tiling 会将 Tensor 切至中等大小（如不超过 512×512），超大的 `MaxOutcastMem()` 通常意味着未经 Tiling 的超大 Tensor 进入了子图。

常见原因：
- **Inplace 操作**：Inplace Tensor 不参与 Tiling 切分
- **Fixed Address Tensor**：固定地址 Tensor 保持原始大小
- **shmemData**：共享内存数据
- **其他框架未正确处理的例外 case**

→ 进入步骤 6 定位具体的问题 Tensor。

---

### 步骤 6：定位问题 Tensor

提取超大 Tensor 告警日志：

```bash
grep -r "\[workspaceSize\].*staticMemReq=\|staticMemReq=.*too larger" <log_path>/debug/
```

**期望日志格式**：

```
[workspaceSize] Root=[<ROOT_FUNC_NAME>], symbol=[<TENSOR_SYMBOL>],rawmagic=[<RAWMAGIC>]: staticMemReq=[<SIZE>] is too larger, which might indicate an error
```

**分析要点**：

1. **匹配异常大小**：用步骤 5 中确定的异常值（如 `MaxOutcastMem() = C`）在告警日志中搜索 `staticMemReq=[C]`，定位匹配的 Tensor
2. **记录关键信息**：
   - `ROOT_FUNC_NAME`：Root Function 名称
   - `TENSOR_SYMBOL`：Tensor 符号名（可能为空）
   - `RAWMAGIC`：Tensor 的 rawmagic 标识
   - `SIZE`：静态内存大小
3. **结合 Root Function 名称**：确认该 Tensor 是否位于步骤 5 中识别的问题 Root Function 内

**⚠️ 注意**：
- 动态 shape 场景下 `maxStaticMemReq` 为 0，不会出现在告警中
- 匿名 Tensor 的 symbol 为空，需通过 rawmagic 在计算图中定位
- 可能存在多个超大 Tensor，需逐一分析

---

### 步骤 7：结合计算图确认问题位置

若步骤 2 中已打开 `dump_graph`，则 `build/output/pass/` 目录下会有各 pass 阶段的计算图文件。

1. **在计算图中搜索目标 Tensor**：
   根据步骤 6 获取的 `ROOT_FUNC_NAME` 找到对应计算图文件，使用 `RAWMAGIC` 值搜索该 Tensor 节点

2. **分析 Tensor 上下游操作**：
   确认该 Tensor 的 shape 来源，重点关注：
   - 是否为 Inplace 操作的输出
   - 是否为 Fixed Address Tensor
   - 是否涉及 reshape / assemble / view 操作
   - 该 Tensor 的 shape 是否与预期一致

3. **确定问题归属**：
   - **用例写法问题**：检查用例中 API 使用是否满足约束
   - **框架侧问题**：联系 machine/pass 同事进一步分析

---

### 步骤 8：输出诊断报告

根据以上分析结果，生成结构化诊断报告。

**报告模板**：

```markdown
## Workspace 内存异常诊断报告

### 基本信息
- 报错类型：<torch OOM / rtMalloc failed / 其他>
- workspace 总大小：<TOTAL> bytes（<换算为 MB 或 GB>）
- 测试命令：<test_cmd>

### Workspace 大小构成

| 子项 | 大小 (bytes) | 占比 |
|------|-------------|------|
| tensor | <T> | <T/TOTAL %> |
| metadata | <M> | <M/TOTAL %> |
| aicoreSpilled | <S> | <S/TOTAL %> |
| debug | <DD + LD> | <占比 %> |

### Tensor Workspace 拆分

| 子项 | 大小 (bytes) | 占比 |
|------|-------------|------|
| rootInnerSpilledMem | <A> | <A/T %> |
| devTaskInnerOutCasts | <B> | <B/T %> |
| BoundaryOutcast (C×D) | <C×D> | <C×D/T %> |

- MaxOutcastMem (单体大小): <C>
- devTaskBoundaryOutcastNum: <X>, devTaskInnerTemporalOutcastNum: <Y>

### 问题定位结果
- 异常子项：<rootInnerSpilledMem / devTaskInnerOutCasts / BoundaryOutcast / 其他>
- 问题 Root Function：<ROOT_FUNC_NAME>
- 问题 Tensor：symbol=<TENSOR_SYMBOL>, rawmagic=<RAWMAGIC>, staticMemReq=<SIZE>
- 原因分析：<具体分析>

### 建议
1. <配置调优建议 / 联系相关组件同事 / 检查用例写法>
2. <后续步骤>
```

---

## 补充定位手段

当上述标准流程无法定位问题时，可使用以下补充手段：

### 补充手段 A：运行时分配失败定位

若报错信息为运行时分配失败（而非编译期预算过大），搜索以下关键错误信息：

```bash
# 运行时 rootInnerSpilledMem 分配失败
grep -r "cannot allocate root inner workspace" <log_path>/debug/

# 通用分配失败
grep -r "Memory not enough" <log_path>/debug/

# SeqWsAllocator 分配失败
grep -r "SeqWsAllocator cannot allocate" <log_path>/debug/
```

这类报错说明编译期预算不足以满足运行时实际需求，需对比编译期预算值与运行时实际请求值。

### 补充手段 B：metadata 内存异常定位

metadata 内存异常的典型报错：

```
Memory not enough(alloc <SIZE>), WsProperty:metadata, WsAddr:<ADDR>, WsSize:<TOTAL>, AllocatedCnt:<USED>, ResetTimes:<N>
workspace.init.check: Slab alloc null,type=<T>,objsize=<S>
```

提取元数据子项日志：

```bash
grep -r "ItemPoolMemSize\|vectorMemSize\|slotAllocatorMemSize\|generalMetadataSlotSize\|MetadataSlabSize\|RequiredSlabNum" <log_path>/debug/
```

metadata 类问题涉及运行时元数据内存管理，建议直接联系 machine 同事分析。

### 补充手段 C：扩大 workspace 快速验证

在 `python/pypto/frontend/parser/entry.py` 中修改 workspace 分配大小：

```python
# 原始
workspace_tensor = torch.empty(workspace_size, dtype=torch.uint8, device=device)
# 修改为
workspace_tensor = torch.empty(workspace_size * 10, dtype=torch.uint8, device=device)
```

若问题不复现，确认为 workspace 预算计算问题。

### 补充手段 D：workspace 改为内部自管理

在 `framework/src/machine/runtime/device_launcher.h` 中，将 `config.workspaceAddr` 的条件分支屏蔽：

```cpp
// 原始
if (config.workspaceAddr) {
// 修改为
if (0) {
```

若问题不复现，说明存在 workspace 内存踩踏等使用层面的问题。

---

## 关键注意事项

1. **日志版本**：`[workspaceSize]` 前缀的日志为统一新增，需确保使用包含该日志的 whl 包版本；若日志中不存在此前缀，可搜索不带前缀的等价日志
2. **动态 shape**：动态 shape 场景下 `maxStaticMemReq` 为 0，超大 Tensor 告警不适用，需结合其他手段
3. **信息收集**：第一步必须通过 `question` 工具收集信息，严禁猜测
4. **停止条件**：若在步骤 3 中无法提取到任何 workspace 相关日志，说明日志未正确生成，需回到步骤 2 重新配置
5. **路径规范**：所有路径必须使用绝对路径

## 参考文档

| 文件 | 说明 |
|------|------|
| `docs/zh/trouble_shooting/machine.md` | MACHINE 组件错误码与排查建议（含 Workspace 内存异常偏大章节） |
| `framework/src/machine/utils/dynamic/dev_encode_program.h` | memBudget 结构体定义 |
| `framework/src/machine/utils/dynamic/dev_encode.cpp` | Workspace 预算计算核心逻辑（CalcTensorWorkspace、EncodeRawShape 等） |
| `framework/src/machine/runtime/device_launcher.h` | Workspace 分配与总量日志打印 |
| `framework/src/machine/utils/dynamic/dev_workspace.h` | 运行时 workspace 分配器（InitTensorAllocators 等） |
| `framework/src/machine/utils/dynamic/allocator/seq_ws_allocator.h` | 顺序分配器与 "Memory not enough" 断言 |
