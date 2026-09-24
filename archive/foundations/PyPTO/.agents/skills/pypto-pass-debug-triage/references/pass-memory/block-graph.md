# Block Graph

## InferParamIndex

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | InferParamIndex |
| 所属目录 | `framework/src/passes/block_graph_pass/` |
| 主要源文件 | `infer_param_index.cpp`、`infer_param_index.h` |
| Pipeline 阶段 | `PVC2_OOO` 策略，block graph 阶段 |
| 前置依赖 Pass | `SubgraphToFunction` |
| 后置消费 Pass | `SrcDstBufferMerge`、`AddAlloc`、`OoOSchedule` |
| 对应 bug 模式 | S001、S002、S003、S005、S009、S015、S035 |

---

### 2. 设计目标

在 `SubgraphToFunction` 之后，重置并重新推导 copy-in/copy-out、view、assemble 及标号维算子的动态 valid shape，然后将每个符号维度注册到 function 的 `DynParamTable`，供运行时通过 COA 索引绑定。

---

### 3. 核心不变量

- `ResetDynValidShape` 仅对 `setSymDimOps` 与 copy-in/out 清除 dynValidShape；`useSelfOps`（ASSEMBLE、L0C_COPY_UB、VIEW）保持不变。
- `ResetGmCopyDynValidShape` 需区分 copy-in/copy-out 及 `isDistCopyOut`。
- `tensorBaseAddrCoaIndex`（来自 `GetIOpAttrOffset`/`GetOOpAttrOffset`）非 -1 时才注册。
- 同一 symbol 在同一 subFunc 中只能对应一个 `DynParamInfo`。
- `InferShape` 通过 `TopoProgramUtils::TopoProgram` 按拓扑顺序执行。

---

### 4. 关键设计决策与取舍

| 决策 | 选择 | 原因 | 如果改这里要注意什么 |
|------|------|------|---------------------|
| 先重置再推导 | 清空 dynValidShape 后跑 InferShape | 避免 tensor graph 阶段残留旧值 | 必须保留 self-use ops 不变 |
| COA 索引公式 | `GetCoaIndex` 与 runtime 布局一致 | 符号维与 caller arg list 对齐 | 修改需同步 interpreter |
| 按 baseAddrCoaIndex 分组 | `addr2ValidShape` map | 同一 tensor 地址的维度聚合 | 不同 tensor 共享 base index 会冲突 |

---

### 5. 已知脆弱场景

- **S035**：数组/vector 越界。触发点：`InsertAddr2ValidShapeSpecified` 访问 `op.GetOOperands()[i]` 与 `op.GetIOperands()[0]` 未校验 size；`SetSubValidShape` 索引 `addr2ValidShapeSpecified[tensorBaseAddrCoaIndex][dimIdx]` 未校验。检查点：构造输入输出数不一致的 copy op。典型报错：`std::out_of_range`。
- **S001**：空 op 列表。触发点：`InferShape` 在 `opList.empty()` 时直接返回 FAILED。检查点：空 sub-function。典型报错：“There is no operation in function”。
- **S003**：`std::static_pointer_cast<CopyOpAttribute>` 未判空。触发点：`InsertAddr2ValidShapeSpecified` 中 copy attr 转换。检查点：缺失 attribute 的 copy op。典型报错：bad_cast。
- **S009**：动态 validshape 丢失。触发点：`ResetOutputDynValidShape` 用 `"sym_" + magic + "_dim_"` 构造符号维，若后续推导失败则无法绑定。检查点：dump `DynParamTable`。典型报错：EMPTY_VALIDSHAPE。
- **S015**：逻辑倒置。触发点：`shouldUpdateDynValidShape = (!function.IsFromOutCast(outOperand) || distCopyType)`，当 `distCopyType` 为 null 时行为可能出乎意料。检查点：dist copy 与 outcast 组合。典型报错：outcast validshape 被误更新。
- **S005**：内存类型推断过窄。触发点：`ResetGmCopyDynValidShape` 仅按 DDR 判断 GM spill。检查点：L0C/L1 GM copy。典型报错：copy 方向错误。

---

### 6. 常见被误判为 Pass 问题的症状

| 症状 | 实际常见根因模块 | 快速验证方法 |
|------|------------------|--------------|
| 符号维度运行时未绑定 | interpreter / operation（N045、N028） | 检查 `DynParamTable` 与 caller `linearArgList` 长度 |
| Outcast validshape 错误 | codegen / machine（N066、N034） | 检查生成 reshape copy 的 valid shape |
| 分布式 copy validshape 不匹配 | distributed / operation（N022、N032） | 验证 `isDistCopyOut` 与 shmem signal buffer size |
| 空 dynValidShape | `SubgraphToFunction` / `InferDynShape` | Dump 前后 `SubgraphToFunction` 与 `InferDynShape` 输出 |

---

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`。

本词条补充检查项：

- [ ] 修改 `GetCoaIndex` 后是否同步 runtime interpreter 的索引公式？

---

### 8. 调试快速入口

- 开启 Pass dump：

  ```python
  pypto.set_pass_config("PVC2_OOO", "InferParamIndex",
                        pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
  ```

- 关键中间状态：`addr2ValidShape`、`addr2ValidShapeSpecified`、`DynParamTable`（`DumpParamIndex`）、`opList` 拓扑顺序。
- 推荐先跑的 UT：无专属 UT；可跑 `framework/tests/ut/passes/src/test_subgraph_to_function.cpp` 及 PVC2_OOO 集成用例。
- 相关 checker：无专属 checker；可配合 `framework/src/passes/pass_check/pre_graph_checker.cpp`、`framework/src/passes/pass_check/assign_memory_type_checker.cpp` 使用。

---

---

## AddAlloc

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | AddAlloc |
| 主要源文件 | `framework/src/passes/block_graph_pass/schedule_ooo/pre_schedule/add_alloc.{h,cpp}` |
| Pipeline 阶段 | block_graph；OoO pre-schedule，`OoOSchedule` 之前 |
| 前置依赖 Pass | block graph 构建与前序准备 Pass |
| 后置消费 Pass | `OoOSchedule` |
| 对应 bug 模式 | C001、S001、S002 |

### 2. 设计目标

在 OoO 调度前插入并规范化内存分配语义，使调度器获得完整的 buffer 生命周期信息。

### 3. 核心不变量

- 每个需要本地 buffer 的对象都有唯一且可追踪的 alloc 生命周期。
- 插入 alloc 不改变原计算数据依赖。

### 4. 关键设计决策与取舍

该 Pass 是结构性步骤，不能通过禁用来隔离；应检查 alloc 列表和调度器输入。

### 5. 已知脆弱场景

- 重复 alloc 或遗漏多输出 tensor。
- 生命周期边界与 producer/consumer 顺序不一致。

### 6. 常见被误判为 Pass 问题的症状

buffer 过大也可能来自上游 tile shape；先区分“分配条目错误”和“尺寸输入错误”。

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`；额外核对 alloc 生命周期与所有输出。

### 8. 调试快速入口

- UT：`framework/tests/ut/passes/src/test_add_alloc.cpp`。
- Dump `OoOSchedule` 前的 alloc op、buffer size 与使用区间。

---

## SpillBuffer

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | SpillBuffer |
| 所属目录 | `framework/src/passes/block_graph_pass/schedule_ooo/post_schedule/` |
| 主要源文件 | `spill_buffer.cpp`、`ooo_scheduler.h`（`OoOScheduler` 成员函数实现） |
| Pipeline 阶段 | block_graph（内嵌于 `OoOSchedule` Pass） |
| 前置依赖 Pass | `OoOSchedule::Init`、`AddAlloc` 已插入 ALLOC op；`AssignMemoryType` 已确定 memory type 与 dtype |
| 后置消费 Pass | `OoOSchedule::SeqSchedule` / `ScheduleMainLoop`、`TuneTileOpSeqForVF`、`RemoveAlloc`、`CopyOutResolve`、`InsertSync` |
| 对应 bug 模式 | P002、P003、P006、P011、S001、S004、S009、S010、S011、C001 |
| 调试开关策略 | `SpillBuffer` 是 `OoOSchedule` 内部必要逻辑，不建议通过关闭 OoO schedule 定界；优先开启 dump、health_check 和 schedule checker |

---

### 2. 设计目标

在 `OoOScheduler` 顺序模拟阶段，当某个 ALLOC 请求无法放入本地 buffer 池（UB/L1/L0C）时，把已占用但未退休的 tensor 搬运到 DDR workspace，并插入 `COPY_OUT` / `COPY_IN` / 新 ALLOC 等 ops，使得后续分配能够成功。

---

### 3. 核心不变量

- 不变量 1：spill 后原 tensor 的 consumer 必须通过新 COPY_IN 的 memId 重新连到新 tensor（`UpdateOperationInput` 按 memId 匹配）。
- 不变量 2：`SpillContext` 中 `newCopyoutOps`、`newAllocOps`、`deleteAllocOps` 必须被 `ApplySpillContext` 正确消费，且 `numTotalIssues` 需按 `newNotRetiredCopyOutSize - deleteRetiredOpSize` 调整。
- 不变量 3：spill 产生的 GM tensor 必须注册 workspace 偏移（`CreateGMTensor` 设置 `workspaceBaseOffset` 并 `EmitInitDDRBuffer`）。
- 不变量 4：多 producer / L0C / 3510 L1 等特殊场景的 spill 路径（`SpillMultiProducerBuffer`、`SpillL0CBuffer`、`SpillL1BufferFor3510`）必须保持 dtype 一致。
- 不变量 5：`bufRefCount_` 在删除旧 op 后必须归零（`RemoveSmallShapeSpillResources`）。

---

### 4. 关键设计决策与取舍

| 决策 | 选择 | 原因 | 如果改这里要注意什么 |
|------|------|------|---------------------|
| spill 选组策略 | `SelectSpillBuffers` 优先选“下次使用时间最晚”的组，兜底 spill all | 最大化腾出空间并减少重复 spill | 修改 `GetGroupNextUseTime` / `GetBufNextUseTime` 要注意 dualdst 双池同地址约束 |
| L0C spill 按 dtype 分组 | `SpillL0CBuffer` 把 consumer 按输出 dtype 分组分别 COPY_IN | reload target 的 dtype 可能与源不同，需在 copyOut 路径完成转换（P006） | 新增 dtype 场景必须验证 `CreateCopyinOp` 的 dtype 是否正确 |
| 多 producer 场景 | `SpillMultiProducerBuffer` 生成 assemble 替代原 tensor | 避免断开多个 producer 的数据流 | 必须同步更新 `UpdateRemainMemid` 与 `UpdateSpillOpDepend`（P011） |

---

### 5. 已知脆弱场景

- **模式 P002**：`IsUnusedTensor` 语义与命名不符（历史上 `HasUnexecutedProducer`）。触发条件：小 shape spill 删除路径误判 tensor 是否还有未执行 consumer。检查点：`IsUnusedTensor` 返回值与调用处 `UpdateNeedDeleteScheduleStatus`。典型报错：consumer 输入被错误删除导致断图。
- **模式 P003**：统计字段 `deleteRetiredOpSize` 被覆盖而非累加。触发条件：`RemoveSmallShapeSpillResources` 被调用多次。检查点：`ctx.deleteRetiredOpSize += deleteNum`。典型报错：PC 计数错误，后续指令错位。
- **模式 P006**：spill GM buffer 的 dtype 与 reload target 不一致。触发条件：L0C consumer 需要非源 dtype。检查点：`SpillL0CBuffer` 中 `dtypeGroups` 分组与 `CreateGMTensor` 的 dtype。典型报错：运行时 MTE dtype 转换失败。
- **模式 P011**：多 producer / 多 consumer 场景 rewire 错误。触发条件：spill tensor 有多个 producer 或 consumer。检查点：`UpdateOperationInput` 按 memId 匹配、`UpdateSpillOpDepend`。典型报错：其他输出被错误替换。
- **模式 S010**：`size_t` 下溢或统计字段覆盖。触发条件：`SpillContext` 中 `deleteNotRetiredOpSize`、`newNotRetiredCopyOutSize` 计算。检查点：`ApplySpillContext` 第 220 行 `numTotalIssues += ...`。
- **模式 S011**：`UpdateNeedDeleteScheduleStatus` 等返回值未检查。触发条件：小 shape spill 路径。检查点：所有 `UpdateNeedDeleteScheduleStatus` 调用是否用 `!= SUCCESS` 判断。

---

### 6. 常见被误判为 Pass 问题的症状

| 症状 | 实际常见根因模块 | 快速验证方法 |
|------|------------------|--------------|
| `Spill all buffer failed` / `Select buffer to spill failed` | `SetHeuristicTileShapes` 产生过大 tile 或 `AddAlloc` 未正确插入 alloc | 查看 `localBufferMap_[memId]->size` 与 pool 总大小，确认是否单 tensor 已接近上限 |
| 运行时 workspace 地址冲突或 OOM | `GlobalMemoryReUse` / machine 层 workspace 分配 | 检查 `workspaceOffset` 与 `ddrKindMap_` 中 SPILL_TEMP 区间是否有重叠 |
| COPY_IN / COPY_OUT dtype 不匹配 | `AssignMemoryType` 对 L0C/UB 的 dtype 路径推断错误 | 对比 `CreateGMTensor` dtype 与 consumer 输出 dtype |
| 多输出 op 的某个输出消失 | `RemoveRedundantOp` / `MixSubgraphSplit` 提前断边 | 在 spill 前后分别 dump graph，确认 producer/consumer 关系 |

---

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`。

---

### 8. 调试快速入口

- 开启 Pass dump 的命令/环境变量：在 `tile_fwk_config.json` 中为 `OoOSchedule` 配置 `"dump_graph": true`、`"health_check": true`；`global.log_level` 调至 DEBUG 可查看 `APASS_LOG_DEBUG_F` 的 spill 过程。
- 关键中间状态：`localBufferMap_`、`tensorOccupyMap`、`bufRefCount_`、`SpillContext` 内容、`orderedOps` 中新增/删除的 ops。
- 推荐先跑的 UT：`framework/tests/ut/passes/src/test_schedule_ooo.cpp`。
- 相关 checker：`framework/src/passes/pass_check/schedule_ooo_checker.cpp`、`framework/src/passes/pass_check/schedule_ooo_checker.h`。

---

---

## OoOScheduler

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | OoOScheduler |
| 所属目录 | `framework/src/passes/block_graph_pass/schedule_ooo/post_schedule/` |
| 主要源文件 | `ooo_scheduler.cpp`、`ooo_scheduler.h`；被 `schedule_ooo.cpp` 中的 `OoOSchedule` Pass 调用 |
| Pipeline 阶段 | block_graph（`OoOSchedule` Pass 的核心调度器） |
| 前置依赖 Pass | `AddAlloc`（已插入 ALLOC）、`AssignMemoryType`（memory type/dtype 已定）、`GraphPartition` / `SubgraphToFunction`（已划分子图） |
| 后置消费 Pass | `TuneTileOpSeqForVF`、`RemoveAlloc`、`CopyOutResolve`、`InsertSync`、`TuneSyncForVF`、`MixSubgraphSplit`、`GlobalMemoryReUse` |
| 对应 bug 模式 | P002、P006、P011、P014、S001、S010、C001 |
| 调试开关策略 | 必要调度链路；不要建议关闭 `OoOSchedule` / `OoOScheduler`。怀疑该模块时用 dump、health_check、pre/post checker 和前后图不变量对比 |

---

### 2. 设计目标

对已划分的 block 子图进行乱序执行模拟，确定每个 op 的 `execOrder`、core 归属（`CoreLocationType` / `AIVCore`）、pipe 类型以及本地 buffer 的生命周期，并生成 spill 指令与 workspace 分配，最终输出一个可被后端发射的 op 序列。

---

### 3. 核心不变量

- 不变量 1：`orderedOps` 中 op magic 与 `schedInfoMap_` key 唯一，且 `newOperations_` 去重（`GetNewOperations`）。
- 不变量 2：每个 ALLOC op 的输出 tensor 必须被 `BufferPool::Allocate` 成功，否则必须触发 spill（`ExecuteAllocIssue`）。
- 不变量 3：op 退休时 `bufRefCount_` 必须归零，否则报 `Tensor[X] bufRefCount not equal to 0`。
- 不变量 4：`issueQueues` / `allocIssueQueue` 必须按 `execOrder` 维护堆序，避免乱序发射。
- 不变量 5：`depManager_` 必须在 spill/ dualdst fuse 等改图后重新初始化（`InitDependencies`）。

---

### 4. 关键设计决策与取舍

| 决策 | 选择 | 原因 | 如果改这里要注意什么 |
|------|------|------|---------------------|
| 顺序模拟 + 乱序发射两阶段 | `SeqSchedule` 先生成 spill，`ScheduleMainLoop` 再乱序执行 | 在乱序前解决内存冲突，简化生命周期计算 | 修改 `SeqSchedule` 与 `ScheduleMainLoop` 的边界必须同步更新 `ApplySpillContext` |
| AIV0/AIV1 双池 | Mix 图启用 `CORE_INIT_CONFIGS_HARDWARE_TWO` | 支持 vector 双核并行 | dualdst 场景下 `ResolveCoreForFree` 必须查询 `dualDstMemIdCoreOverride_` |
| 退休唤醒后继 | `RetireOpAndAwakeSucc` 中遍历 `depManager_.GetSuccessors` | 保证数据依赖 | 修改 pred/retired 条件时要注意 `CheckAndUpdateLifecycle` 的校验 |

---

### 5. 已知脆弱场景

- **模式 P014**：branch / event id 使用错误。触发条件：Mix 图中 vector/Cube op 的 `vecBranchId` 或 `pipeType` 分配错误。检查点：`RescheduleUtils::GetOpPipeType`、`InitOpCoreType`。典型报错：同步事件超限或执行顺序错误。
- **模式 P002**：`HasUnexecutedProducer` 与 `IsUnusedTensor` 语义命名相反。触发条件：spill 小 shape 场景依赖该函数判断。检查点：`ooo_scheduler.h` 中 `IsUnusedTensor` 声明与实现。
- **模式 P006 / P011**：L0C spill 与多 consumer 场景。触发条件：L0C tensor 有多个 UB/L1 consumer 或 dtype 不同。检查点：`SpillL0CBuffer` 分组、`UpdateSuccessorDependencies`。
- **模式 S010**：`numTotalIssues` / `deleteRetiredOpSize` 等无符号下溢或覆盖。触发条件：`ApplySpillContext` 第 220 行 `numTotalIssues += ctx.newCopyoutOps.size() - ctx.deleteNotRetiredOpSize`。
- **模式 C001**：运行时 spill / 同步 / 地址报错。触发条件：machine 层 launch 与 pass 推断不一致。检查点：对比 pass 输出的 `memoryrange` 与 machine 实际分配。

---

### 6. 常见被误判为 Pass 问题的症状

| 症状 | 实际常见根因模块 | 快速验证方法 |
|------|------------------|--------------|
| `Unexecuted op` / `bufRefCount not equal to 0` | `InsertSync` 或 `TuneSyncForVF` 事件依赖缺失 | 检查 `depManager_` 边与 `InsertSync` 生成的事件 id 是否覆盖所有跨子图依赖 |
| `Tensor[X] not in bufferSlices` | `BufferPool` 自由偏移计算错误，或 dualdst 核归属错误 | 查看 `dualDstMemIdCoreOverride_` 与 `ResolveCoreForFree` |
| 性能回退（schedule 后 latency 变大） | `TaskSplitter` / `CoreScheduler` 的 mix 切分策略 | 对比 MixSchedule 与 NonMixSchedule 的 `ScheduleUnit` 起止时间 |
| OOM / workspace 过大 | `GlobalMemoryReUse` 未合并或 `AllocWorkspaceGM` 分配过多 | 检查 `workspaceOffset` 与 GM tensor 数量 |

---

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`。

---

### 8. 调试快速入口

- 开启 Pass dump 的命令/环境变量：`tile_fwk_config.json` 中 `OoOSchedule` 配置 `"dump_graph": true`（会输出 memory trace）、`"health_check": true`（生成 `Block_Graph_Health_Report.json`）。
- 关键中间状态：`schedInfoMap_`（`execOrder`、`coreLocation`、`pipeType`、`isRetired`）、`localBufferMap_`、`tensorOccupyMap`、`issueQueues`、`allocIssueQueue`。
- 推荐先跑的 UT：`framework/tests/ut/passes/src/test_schedule_ooo.cpp`。
- 相关 checker：`framework/src/passes/pass_check/schedule_ooo_checker.cpp`、`framework/src/passes/pass_check/schedule_ooo_checker.h`。

---

---

## TuneTileOpSeqForVF

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | TuneTileOpSeqForVF |
| 所属目录 | `framework/src/passes/block_graph_pass/` |
| 主要源文件 | `tune_tileopseq_for_vf.cpp`、`tune_tileopseq_for_vf.h` |
| Pipeline 阶段 | `PVC2_OOO` 策略，block_graph（`OoOSchedule` 之后） |
| 前置依赖 Pass | `OoOSchedule` |
| 后置消费 Pass | `RemoveAlloc`、`CopyOutResolve`、`InsertSync`、`TuneSyncForVF` |
| 对应 bug 模式 | S001、S008、S035 |

---

### 2. 设计目标

在 `enable_vf` 开启时，重排相邻 PIPE_V 向量 tile op 之间的非 PIPE_V op，使可融合的 vector op 相邻；随后对融合的 group 内部 `OP_UB_COPY_ND2NZ` 进行前后移动，最后根据 producer/consumer 关系重新定位 VIEW/ASSEMBLE。

---

### 3. 核心不变量

- 跳过条件：`config::GetPassGlobalConfig(KEY_ENABLE_VF, false)` 为 false 时直接返回 SUCCESS。
- 仅将 `pipeIdStart_ == PIPE_V` 且 AIV0/AIV1 的 op 作为融合候选。
- `IsMergeable` 拒绝与左右两侧都存在数据依赖的中间 op。
- `OP_VIEW`、`OP_ASSEMBLE`、`OP_NOP`、`OP_HUB` 视为可移动 op。
- UB copy 移动必须保持与 group 内非 UB op 的依赖。
- 最终通过 `ScheduleBy(opList_, true)` 写回。

---

### 4. 关键设计决策与取舍

| 决策 | 选择 | 原因 | 如果改这里要注意什么 |
|------|------|------|---------------------|
| 可移动 op 白名单 | VIEW、ASSEMBLE、NOP、HUB | 这些是 metadata/内存布局 op | 加入计算 op 会改变语义 |
| UB copy 重排 | 根据与前/后非 UB op 的依赖决定前移/后移 | 降低 UB 竞争 | 必须验证依赖 |
| VIEW/ASSEMBLE 顺序修正 | `ProcessViewAssembleOrder` 单独处理 | 融合后保持 metadata op 合法 | 失败会留下 dangling producer/consumer |

---

### 5. 已知脆弱场景

- **S035**：数组/vector 越界。触发点：`MoveOpsForMerge` 使用 `opList_.begin() + left - mergedOps[groupNum].size() + 2`；`CollectGroupIndices` 在循环内重复插入并排序 `groupIndices`。检查点：构造 `mergedOps[groupNum].size() > left + 2` 或 group 仅含 UB copy 的图。典型报错：`std::out_of_range`。
- **S001**：空容器解引用。触发点：`ProcessGroupUbCopyOrder` 中 `nonUbCopyIndices.front()` 在空返回后安全，但 `CollectGroupIndices` 内部循环逻辑复杂。检查点：group 中无非 UB op 的情况。典型报错：空 vector front。
- **S008**：硬编码 opcode。触发点：白名单 `OP_VIEW`、`OP_ASSEMBLE`、`OP_NOP`、`OP_HUB` 与特殊处理 `OP_UB_COPY_ND2NZ`。检查点：新增 metadata opcode。典型报错：遗漏可移动 op。

---

### 6. 常见被误判为 Pass 问题的症状

| 症状 | 实际常见根因模块 | 快速验证方法 |
|------|------------------|--------------|
| VF pair 仍不相邻 | `TuneSyncForVF` / codegen | 检查 `TuneSyncForVF` 是否后续移动 sync 标志 |
| UB copy 依赖被破坏 | codegen / machine 内存模型 | 验证生成 MTE 序列与 `PipeSync` range |
| VIEW 被放到 producer 之前 | 上游 Pass / 图不变量 | 跑 `merge_view_assemble_checker` 后验证 |
| 性能无提升 | codegen / machine schedule | NPU profiling 对比 `enable_vf` 开关 |

---

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`。

本词条补充检查项：

- [ ] 修改可移动 op 白名单后是否验证 `ProcessViewAssembleOrder` 仍能保持合法顺序？

---

### 8. 调试快速入口

- 开启 VF 与 Pass dump：在 `tile_fwk_config.json` 中设置 `pass.enable_vf=true`，并执行：

  ```python
  pypto.set_pass_config("PVC2_OOO", "TuneTileOpSeqForVF",
                        pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
  ```

- 关键中间状态：`opList_`、`mergedOps`、`moveFrontOp`、`pipeVIdx`、`ubCopyIndices`/`nonUbCopyIndices`、`changeMap`。
- 推荐先跑的 UT：`TuneTileopseqForVFTest.*`（`framework/tests/ut/passes/src/test_tune_tileopseq_for_vf.cpp`），重点关注 `TestMergeForTuneTileop`、`TestAdjustUbCopyNd2NzOrder_UbCopyMoveFront`。
- 相关 checker：无专属 checker；可配合 `framework/src/passes/pass_check/schedule_ooo_checker.cpp`、`framework/src/passes/pass_check/merge_view_assemble_checker.cpp` 使用。

---

---

## RemoveAlloc

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | RemoveAlloc |
| 主要源文件 | `framework/src/passes/block_graph_pass/schedule_ooo/post_schedule/remove_alloc.{h,cpp}` |
| Pipeline 阶段 | block_graph；`TuneTileOpSeqForVF` 之后、`CopyOutResolve` 之前 |
| 前置依赖 Pass | `OoOSchedule`、`TuneTileOpSeqForVF` |
| 后置消费 Pass | `CopyOutResolve` |
| 对应 bug 模式 | C001、S001、S002 |

### 2. 设计目标

在调度和 VF 顺序确定后移除仅服务于调度建模的 alloc 标记，同时保留真实数据依赖。

### 3. 核心不变量

- 删除 alloc 后所有计算 op 顺序和 buffer 生命周期仍可由调度结果解释。
- 不得删除仍被下游解析所需的结构信息。

### 4. 关键设计决策与取舍

该 Pass 与 `AddAlloc`/`OoOSchedule` 成对工作，是结构性步骤，不允许以跳过作为最终方案。

### 5. 已知脆弱场景

- 多消费者或 spill 路径仍引用 alloc 标记。
- 删除时未同步 producer/consumer 边。

### 6. 常见被误判为 Pass 问题的症状

若调度结果本身已损坏，应先检查 `OoOSchedule`，而不是只在清理阶段修补。

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`；额外核对 alloc/add-remove 成对关系。

### 8. 调试快速入口

- UT：`framework/tests/ut/passes/src/test_remove_alloc.cpp`。
- 对比 Pass 前后 alloc 数量、数据边和调度顺序。

---

## CopyOutResolve

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | CopyOutResolve |
| 主要源文件 | `framework/src/passes/block_graph_pass/copy_out_resolve.{h,cpp}` |
| Pipeline 阶段 | block_graph；`RemoveAlloc` 之后、`InsertSync` 之前 |
| 前置依赖 Pass | `RemoveAlloc` |
| 后置消费 Pass | `InsertSync`（硬依赖本 Pass） |
| 对应 bug 模式 | P025、C001、S001、S009 |

### 2. 设计目标

解析和规范化 copy-out 路径，为同步插入和 codegen 提供确定的数据搬移关系。

### 3. 核心不变量

- copy-out 合并或复用后 offset 与 `raw_shape` 必须同时更新，避免 P025。
- 外部输出语义和数据依赖不能因 coalescing 改变。

### 4. 关键设计决策与取舍

该 Pass 是 `InsertSync` 的结构性前置；需要隔离 coalescing 时优先使用更窄的功能开关。

### 5. 已知脆弱场景

- COPY_OUT → HUB 级联复用只更新 offset，遗漏 `raw_shape`。
- 特殊 copyout opcode 未被统一处理。

### 6. 常见被误判为 Pass 问题的症状

若 memory type 或 raw tensor 属性在进入本 Pass 前已错误，应回查上游数据路径 Pass。

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`；额外核对 offset、raw_shape 和外部输出使用。

### 8. 调试快速入口

- 当前未发现同名 dedicated UT；这是需要补充的测试缺口。
- 现有 codegen dynamic-strategy 用例覆盖部分路径；应新增 raw_shape/offset 联动回归。
- Dump 合并前后的 copy attribute、offset 和 raw_shape。

---

## InsertSync

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | InsertSync |
| 主要源文件 | `framework/src/passes/block_graph_pass/insert_sync.{h,cpp}` |
| Pipeline 阶段 | block_graph；`CopyOutResolve` 之后、`TuneSyncForVF` 之前 |
| 前置依赖 Pass | 硬依赖 `OoOSchedule`、`CopyOutResolve` |
| 后置消费 Pass | `TuneSyncForVF` |
| 对应 bug 模式 | P023、C001、S001、S008 |

### 2. 设计目标

依据调度顺序和 pipe 依赖插入同步事件，保证跨 pipe 数据依赖且避免死锁。

### 3. 核心不变量

- 每个 wait 都有可达的 set，event ID 生命周期不冲突。
- 放松依赖必须尝试合法方向，遍历过程中不能失效当前容器。

### 4. 关键设计决策与取舍

该 Pass 是结构性步骤且依赖调度结果，不能禁用；使用 checker、事件表和前后图对比定界。

### 5. 已知脆弱场景

- event ID 耗尽时把“不可放松”误判为执行失败。
- 遍历 setPipe 时原地修改集合，导致迭代器失效。

### 6. 常见被误判为 Pass 问题的症状

若 `OoOSchedule` 产生的 op 顺序错误，同步报错只是下游表现，应先验证调度不变量。

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`；额外覆盖 event ID 耗尽、双向放松与容器快照。

### 8. 调试快速入口

- UT：`framework/tests/ut/passes/src/test_insert_sync.cpp`。
- Dump set/wait 事件映射、core pair、pipe 依赖与调度 op 序列。

---

## TuneSyncForVF

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | TuneSyncForVF |
| 所属目录 | `framework/src/passes/block_graph_pass/` |
| 主要源文件 | `tune_sync_for_vf.cpp`、`tune_sync_for_vf.h` |
| Pipeline 阶段 | `PVC2_OOO` 策略，block_graph |
| 前置依赖 Pass | `OoOSchedule`、`TuneTileOpSeqForVF`、`InsertSync` |
| 后置消费 Pass | `MixSubgraphSplit` |
| 对应 bug 模式 | S001、S008、S035、S036 |

---

### 2. 设计目标

在 DAV_3510 且 `enable_vf` 开启时，将相邻 PIPE_V 向量 tile op 之间的 `OP_SYNC_SRC`/`OP_SYNC_DST` 标志向融合对靠拢，减少 pipe 空闲时间，同时保证 PIPE_V 与其他 pipe 之间的依赖正确。

---

### 3. 核心不变量

- 跳过条件：`config::GetPassGlobalConfig(KEY_ENABLE_VF, false)` 为 false 时直接返回 SUCCESS。
- 只处理 `pipeIdStart_ == PIPE_V` 且 `AIVCore` 为 AIV0/AIV1 的 op。
- 可融合区间中间只能有 `OP_SYNC_SRC`/`OP_SYNC_DST`。
- `subGraphFunc->setOpMap`/`waitOpMap` 必须包含每个 flag 对应的目标 tile op。
- 时间戳调整后不能出现负的 pipe 开始时间。
- 最终通过 `ScheduleBy(opList_, true)` 写回 function。

---

### 4. 关键设计决策与取舍

| 决策 | 选择 | 原因 | 如果改这里要注意什么 |
|------|------|------|---------------------|
| 启发式系数 | `vfPrarm = 0.8f` | 经验性 VF 融合收益 | 修改需做性能回归 |
| 按 core 分别调整 | 先 AIV0 再 AIV1 | 两 core 的 PIPE_V 调度独立 | 必须保持跨 core 同步 |
| 贪心两两合并 | 任意一个 flag 有收益就融合 | 最大化融合机会 | 可能产生过长融合组 |

---

### 5. 已知脆弱场景

- **S035**：数组/vector 越界。触发点：`MoveOpsForMerge` 中 `opList_.begin() + vecTileOp0Idx + 2` 与 `opList_.begin() + vecTileOp0Idx - mergedSize + 1` 未做边界校验。检查点：构造 `mergedSize > vecTileOp0Idx` 或 `vecTileOp0Idx` 接近末尾的图。典型报错：`std::out_of_range` / 段错误。
- **S036**：`std::map::operator[]` 默认构造导致空指针。触发点：`subGraphFunc->setOpMap[setFlag]` 与 `subGraphFunc->waitOpMap[waitFlag]` 直接解引用，若 flag 不在 map 中返回 nullptr。检查点：缺少对应 tile op 的 SYNC。典型报错：空指针解引用。
- **S008**：硬编码常量。触发点：`vfPrarm = 0.8f`、latency 取值、默认硬件并发数。检查点：集中配置。典型报错：不同平台性能/正确性差异。
- **S001**：空容器。触发点：`UpdateSetPipeTime` 遍历 `pipeOpMap[pipeX]`，若为空则返回 FAILED。检查点：无 pipe-X op 的图。典型报错：Cannot find ... in pipe oplist。

---

### 6. 常见被误判为 Pass 问题的症状

| 症状 | 实际常见根因模块 | 快速验证方法 |
|------|------------------|--------------|
| VF 融合未生效 | codegen / machine（`enable_vf` 配置） | 确认 `KEY_ENABLE_VF=true`，检查 `device_launcher`/`main_block` 分支 |
| 同步后 hang 或结果错误 | `InsertSync` / machine event 管理（N007、N088、N040） | 检查生成 event ID 与 `AclRtDestroyEvent` |
| Pipe 调度时间出现负值 | `OoOSchedule` / codegen | Dump 前后 `cycleStart/cycleEnd` |
| AIV core 不匹配 | codegen / tileop（N088） | 验证 `GetAIVCore()` 与生成 kernel 一致性 |

---

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`。

本词条补充检查项：

- [ ] 修改 `opList_` 索引计算后是否验证空列表、单元素列表、边界索引？

---

### 8. 调试快速入口

- 开启 VF 与 Pass dump：在 `framework/src/interface/configs/tile_fwk_config.json` 中设置 `pass.enable_vf=true`，或设置 `TILEFWK_CONFIG_PATH`；并执行：

  ```python
  pypto.set_pass_config("PVC2_OOO", "TuneSyncForVF",
                        pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
  ```

- 关键中间状态：`opList_`、`pipeVIdx`、`setFlagList`/`waitFlagList`、`mergedOps`、`pipeOpMap`、`cycleStart/cycleEnd`。
- 推荐先跑的 UT：`TuneSyncForVFTest.*`（`framework/tests/ut/passes/src/test_tune_sync_for_vf.cpp`），重点关注 `TestTuneSyncForVF`、`TestMainProcess`。
- 相关 checker：无专属 checker；可配合 `framework/src/passes/pass_check/schedule_ooo_checker.cpp` 使用。

---

---

## DynAttrToStatic

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | DynAttrToStatic |
| 所属目录 | `framework/src/passes/block_graph_pass/` |
| 主要源文件 | `dyn_attr_to_static.cpp`、`dyn_attr_to_static.h` |
| Pipeline 阶段 | `ExecuteGraph` 策略（execute/block graph） |
| 前置依赖 Pass | `LoopUnroll`（同策略内） |
| 后置消费 Pass | `LoopaxesProc` |
| 对应 bug 模式 | S001、S002、S003、S035、N045、N027、C005 |

---

### 2. 设计目标

遍历 execute graph 的 call op，收集每个 leaf function 的所有 caller 实参，将 leaf 内部依赖 COA 的动态 attribute / 符号维度评估为静态值；当所有 caller 取值一致时改写为 `MAYBE_CONST` COA 宏或直接 concrete 值。

---

### 3. 核心不变量

- `BuildLeafToCaller` 只进入 `TENSOR_GRAPH`、`TILE_GRAPH`、`EXECUTE_GRAPH` 类型的 function。
- 仅处理以 `RUNTIME_COA_GET_PARAM` 开头的 COA 表达式。
- 同一 dynScalar 在所有 caller 的对应 `LinearArgList` 索引处必须是相同 immediate 值，才能转为静态。
- `ReplaceCommonSymbol` 只在 caller arg 索引组一致的 symbol 上标记 base param。
- `BuildParamAddr` 只更新带 `tensorAddr` attr 的 tensor。

---

### 4. 关键设计决策与取舍

| 决策 | 选择 | 原因 | 如果改这里要注意什么 |
|------|------|------|---------------------|
| Per-leaf 评估 | 收集该 leaf 所有 caller 的 `LinearArgList` | 捕获全部 caller 上下文 | 漏 caller 会得到错误静态结论 |
| Branch mode 划分 | DEFAULT / STATIC_CONST / VARIABLE / CONST | 区分全静态、统一非 immediate、每调用常量 | 新增 mode 需同步 COA 宏支持 |
| 向量一致性替换 | `VectorParamConsistencyChecker` | 将重复 COA 索引替换为 base param | 分组错误会破坏 symbol 身份 |

---

### 5. 已知脆弱场景

- **S035**：数组/vector 越界索引。触发点：`TryRemoveDynAttr` 中 `argList[coaIndex]` 未校验 `coaIndex < argList.size()`；`BuildParamAddr` 遍历 operand 列表也未校验索引。检查点：构造 caller arg 长度不一致的图。典型报错：`std::out_of_range` / 段错误。
- **S001**：空容器解引用。触发点：若 `leaf2Caller[leafFunc]` 为空，`BuildNewCoa` 的 `scalarValue` 为空，可能产生默认分支。检查点：无 caller 的 leaf function。典型报错：COA 宏被错误改写。
- **S003**：`std::static_pointer_cast` 未判空。触发点：`TryRemoveDynAttr` 中 `std::static_pointer_cast<CallOpAttribute>(callop->GetOpAttribute())`。检查点：非 call op 误入。典型报错：bad_cast / 空指针。
- **N045**：运行时 `linearArgList` 越界。触发点：`CalculateCoaIndex` 公式与 runtime 不一致，导致 interpreter 访问 `linearArgList` 越界。检查点：对比 `CalculateCoaIndex` 与 `framework/src/interface/interpreter/operation.cpp` 索引逻辑。典型报错：动态 shape COA 索引偏移错误。
- **N027**：Call 类 OP 属性转换缺少校验。触发点：`GetCallee` 假设 op 为 call，若前置校验缺失则 `GetCalleeBracketName` 等会 bad cast。检查点：`FE_ASSERT(IsCall())`。典型报错：bad_cast。

---

### 6. 常见被误判为 Pass 的问题症状

| 症状 | 实际常见根因模块 | 快速验证方法 |
|------|------------------|--------------|
| 动态维度未被转换 | frontend / IR（C005、N016） | 检查前端符号绑定与 IR `SymbolicScalar` 身份 |
| 运行时 COA 索引错位 | interpreter / operation（N045、N027） | 跟踪 `framework/src/interface/interpreter/operation.cpp` 中 `linearArgList` 访问 |
| Maybe-const COA 仍被当作 unknown | machine / runtime（N034、N035） | 检查 runtime evaluator 对 `GET_PARAM_ADDR_MAYBE_CONST` 的处理 |
| 多 caller leaf 得到错误静态值 | operation / frontend shape 绑定 | 确认每个 caller 传入的具体值一致 |

---

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`。

本词条补充检查项：

- [ ] 修改 COA 索引公式后是否同步验证 runtime interpreter 的索引公式？

---

### 8. 调试快速入口

- 开启 Pass dump：

  ```python
  pypto.set_pass_config("ExecuteGraph", "DynAttrToStatic",
                        pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
  ```

  同时需在 `tile_fwk_config.json` 中确认 `pass.enable_vf` 等全局开关符合预期，或设置 `TILEFWK_CONFIG_PATH`。
- 关键中间状态：`leaf2Caller`、`callopArglistOneDim`、`coaIndex`、`scalarValue`、`branchMode`、`dynParamTable`（`ReplaceCommonSymbol` 后）。
- 推荐先跑的 UT：`DynAttrToStaticTest.*`（`framework/tests/ut/passes/src/test_dyn_attr_to_static.cpp`），重点关注 `TestDynExpression`、`EdgeCases`。
- 相关 checker：无专属 checker；可配合 `framework/src/passes/pass_check/expand_function_checker.cpp` 上游与 `framework/src/passes/pass_check/schedule_ooo_checker.cpp` 下游使用。

---

---

## LoopaxesProc

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | LoopaxesProc |
| 所属目录 | `framework/src/passes/block_graph_pass/` |
| 主要源文件 | `loopaxes_proc.cpp`、`loopaxes_proc.h` |
| Pipeline 阶段 | block_graph（`ExecuteGraph` 策略，在 `DynAttrToStatic` 之后） |
| 前置依赖 Pass | `DynAttrToStatic`（`BuildLeafToCaller` 提供 leaf->caller 映射） |
| 后置消费 Pass | 无直接后续 Pass；`loopGroup` / `dynloopGroup` / `loopAxes` / `dynloopAxes` 属性供后端 vector fusion / codegen 使用 |
| 对应 bug 模式 | P001、S001、S015、C002 |

---

### 2. 设计目标

在启用 VF（Vector Fusion）时，根据 op 输出形状的循环轴（loopAxes / dynloopAxes）把相邻且轴相同的 op 编组，并基于地址范围重叠检测在组内做合理切分，为后端的 vector fusion 提供 `loopGroup` 与 `dynloopGroup` 标记。

---

### 3. 核心不变量

- 不变量 1：只有 `SUPPORT_VF_FUSE_OPS` 中的 op 才会被编组（`NeedClearStatus`），不支持的 op 会强制结束当前组。
- 不变量 2：`shape.size()` 与 `dynShape.size()` 必须一致，且 `shape.size() > MIN_SHAPE_DIM` 才会进入编组逻辑。
- 不变量 3：动态循环轴相等判定 `SameDynLoopAxes` 必须在 `dynParamTable` 中比较替换符号或表达式（不能仅比较字面量）。
- 不变量 4：地址重叠检测 `CheckAddrOverLap` 发现冲突后，必须通过 `FindCuts` 在冲突边界处切分组。
- 不变量 5：组结束时必须通过 `SetOpLoopEnd` / `SetOpDynLoopEnd` 标记最后一个 op。

---

### 4. 关键设计决策与取舍

| 决策 | 选择 | 原因 | 如果改这里要注意什么 |
|------|------|------|---------------------|
| 静态/动态循环轴双轨处理 | `ProcessStaticLoopGroup` 与 `ProcessDynLoopGroup` 分别维护 groupIdx 与状态 | 动态轴涉及符号表达式，不能与静态轴混用 | 修改 `SameDynLoopAxes` 必须同步维护 `dynPreviousLoopAxes` 与 `dynParamTable` 语义 |
| 地址冲突按 input/output 独立检查 | `IsOverLap` 分别检查 `noOverlapWithInput` 与 `noOverlapWithOutput` | 避免把输入与输出范围交叉比较导致漏判（P001） | 修改 overlap 逻辑必须同时覆盖相邻但不重叠、部分重叠两类 adversarial case |
| 暴力枚举切分点 | `FindCuts` 对 groupSize-1 个位置做 `1 << (groupSize-1)` 枚举 | group 通常很小，可接受 | 新增极大 group 场景要考虑组合爆炸风险 |

---

### 5. 已知脆弱场景

- **模式 P001**：`IsOverLap` 原来把输入范围与输出范围交叉比较。触发条件：同一组内既有输入范围冲突又有输出范围冲突。检查点：`IsOverLap` 中 `noOverlapWithInput` / `noOverlapWithOutput` 是否独立。典型报错：运行时地址冲突或 VF 结果错误。
- **模式 S015**：`SameDynLoopAxes` 命名/语义可能误导（返回 true 当且仅当符号或表达式匹配）。触发条件：动态 shape 下 loop 组合并或切分错误。检查点：函数返回值与调用方预期是否一致。
- **模式 S001**：`GetOpLoopAxes`、`UpdateOpLoopAxes` 中直接取 `GetOOperands().front()` 未判空。触发条件：异常 op 无输出。检查点：`GetOOperands().empty()` 前置检查。
- **模式 C002**：`dynloopAxes` 来自 `DynValidShape`，若上游 `InferDynShape` / `DynAttrToStatic` 推导错误，本 Pass 会在错误 shape 上切分组。检查点：dump 本 Pass 输入的 `dynShape` 与 `validshape`。

---

### 6. 常见被误判为 Pass 问题的症状

| 症状 | 实际常见根因模块 | 快速验证方法 |
|------|------------------|--------------|
| VF 融合结果错误 / 精度异常 | `TuneTileOpSeqForVF` / `TuneSyncForVF` 的同步或 tile 序列错误 | 关闭 VF（`enable_vf=false`）后若错误消失，再检查 `loopGroup` 标记 |
| 后端报 loop 轴不匹配 | `InferDynShape` / `DynAttrToStatic` 对 leaf function 的 dynParam 映射错误 | 检查 `DynAttrToStatic::BuildLeafToCaller` 输出与 `dynloopAxes` 是否一致 |
| 地址冲突但 LoopaxesProc 未切分 | `AssignMemoryType` / `OoOScheduler` 给出的 `memoryrange` 与编组时使用的范围不一致 | 对比 `RecordAddrOverLap` 中读取的 `memoryrange.start/end` 与 pass 后实际分配 |
| 不支持 VF 的 op 被错误编组 | `SUPPORT_VF_FUSE_OPS` 集合与前端/operation 注册不一致 | 查看 `NeedClearStatus` 中 opcode 判断 |

---

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`。

---

### 8. 调试快速入口

- 开启 Pass dump 的命令/环境变量：`tile_fwk_config.json` 设置 `global.pass.default_pass_configs.dump_graph=true`、`print_graph=true`；本 Pass 日志关键字 `MODULE_NAME "LoopaxesProc"`。
- 关键中间状态：`loopAxes`、`dynloopAxes`、`groupIdx`、`addrStaticRecordMap` / `addrDynRecordMap`、`addrStaticConflictIdx` / `addrDynConflictIdx`、`FindCuts` 结果。
- 推荐先跑的 UT：`framework/tests/ut/passes/src/test_loopaxesproc.cpp`。
- 相关 checker：无专属 checker；通用校验见 `framework/src/passes/pass_check/checker.h`。

---

---

## CodegenPreproc

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | CodegenPreproc |
| 主要源文件 | `framework/src/passes/block_graph_pass/codegen_preproc.{h,cpp}` |
| Pipeline 阶段 | block_graph → codegen 边界；`PVC2_OOO` 默认策略末项 |
| 前置依赖 Pass | 前序 block_graph Pass；无额外显式 dependency 注册 |
| 后置消费 Pass | codegen |
| 对应 bug 模式 | C007、S004、S006、S009 |

### 2. 设计目标

在进入 codegen 前规范化 block graph 与属性，使后端只接收满足约束的最终表示。

### 3. 核心不变量

- codegen 所需的 shape、dtype、memory type、offset 与属性全部可用且一致。
- 预处理不能改变计算语义或掩盖上游非法图。

### 4. 关键设计决策与取舍

这是 codegen 边界的结构性 Pass；失败时应区分上游图非法和预处理实现缺陷，不能建议禁用。

### 5. 已知脆弱场景

- 新 opcode/属性未纳入预处理分支。
- 动态 shape 或对齐约束在进入 codegen 前未固化。

### 6. 常见被误判为 Pass 问题的症状

后端 unsupported 报错也可能是 operation 或上游 memory type 推导错误；先检查本 Pass 的输入。

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`；额外核对 codegen 支持矩阵和新 opcode。

### 8. 调试快速入口

- UT：`framework/tests/ut/passes/src/test_codegen_preproc.cpp`。
- 对比本 Pass 前后 block graph，并核对 codegen 输入属性。

---
