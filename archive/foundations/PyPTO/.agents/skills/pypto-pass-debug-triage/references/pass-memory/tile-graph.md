# Tile Graph

## SplitLargeFanoutTensor

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | SplitLargeFanoutTensor |
| 所属目录 | `framework/src/passes/tile_graph_pass/graph_optimization/` |
| 主要源文件 | `split_large_fanout_tensor.cpp`、`split_large_fanout_tensor.h` |
| Pipeline 阶段 | `PVC2_OOO` 策略，tile/block graph 阶段 |
| 前置依赖 Pass | `MergeViewAssemble`、`SplitReshape`、`SplitRawTensor` |
| 后置消费 Pass | `DuplicateOp`、`AssignMemoryType`、`InferDiscontinuousInput` |
| 对应 bug 模式 | P005、S001、S002、S003、S004、S005、S006、S009、S012、S015、S017 |

---

### 2. 设计目标

识别 `OP_ASSEMBLE → largeTensor → OP_VIEW` 的大张量场景，利用 LCM tile 将大 tensor 拆成更小的 tile，使下游 `OP_VIEW` 不必等待完整 assemble 完成即可开始；随后删除冗余的 assemble/view，并对新增算子重新推导 shape/validshape。

---

### 3. 核心不变量

- 只处理生产者 opcode 为 `OP_ASSEMBLE` 且消费者 opcode 为 `OP_VIEW` 的 tensor（`CollectLargeTensor`）。
- 所有 assemble 输入（`toInfoMap_`）的 offset 必须落在大 tensor 内且彼此不同（`IsBeCovered`、`HasDuplicateToTile`）。
- 每个 LCM tile 必须被收集到的 overlap 完全覆盖（`CheckOverlapCoverage`）。
- 新建 tensor 必须继承原 tensor 的 `Datatype`、`Format`，并保留动态 `dynOffset`/`validShape`。
- 删除冗余 assemble/view 后，producer/consumer 关系必须保持一致（`UpdateForRedundantAssemble`、`UpdateForRedundantView`）。
- `addedOps_` 中的新算子必须通过 `InferShapeUtils::InferShape` 完成 shape 推导。

---

### 4. 关键设计决策与取舍

| 决策 | 选择 | 原因 | 如果改这里要注意什么 |
|------|------|------|---------------------|
| Tile shape 选择 | 逐维取 assemble 输入 shape 与 view 输出 shape 的 LCM，并以 largeTensor shape 为上限 | 保证两边在 tile 边界对齐 | 改为 GCD 或 max 会破坏覆盖性 |
| 覆盖性校验 | 要求 overlaps 面积之和等于 lcmTile 面积 | 防止部分 tile 缺数据 | 放宽会产生静默数据损坏 |
| 动态 offset 处理 | 当 view 存在非 concrete 动态 offset 时跳过 | 无法确定会用到哪些 assemble tile | 支持动态拆分需引入符号边界 |
| 冗余 view 折叠 | 仅当所有子节点都是 `OP_VIEW` 且不是 L1 多加载场景时才删除 | 避免误删 tiling 为 L1 copy 插入的 view | 漏判 L1 场景会导致 memory type 错误 |

---

### 5. 已知脆弱场景

- **P005**：view 合并后 validshape 推导错误。触发点：`UpdateForRedundantView` 使用 `GetViewValidShape` 与 consumer 的 `ToDynValidShape` 推导，若源 tensor 与目标动态 shape 没对齐会出错。检查点：dump `EraseRedundantViewOp` 前后的 validshape，确认 `ToDynValidShape` 来自 op attribute。典型报错：validshape mismatch / merged view size error。
- **S001**：空容器解引用。触发点：`CreateOpFor1toM` 中 `auto viewOp = *dualOverlap->GetProducers().begin();` 未判空。检查点：搜索 `.front()`/`.begin()`/`[0]`，补充 `empty()` 保护。典型报错：segmentation fault / nullptr dereference。
- **S002**：多生产者/消费者只取第一个。触发点：`CollectLargeTensor` 仅取第一个 producer/consumer 判断 opcode；`CreateOpFor1toM` 默认第一个 producer 是 view。检查点：构造共享 tensor 或多个 assemble producer 的图。典型报错：错误跳过或数据流断裂。
- **S003**：类型转换未判空。触发点：`FilterOverlaps` 对 `AssembleOpAttribute`/`ViewOpAttribute` 做 `dynamic_cast` 后未统一判空。检查点：所有 dynamic_cast 结果必须检查。典型报错：bad_cast / 空指针解引用。
- **S006**：整数溢出。触发点：`LCM` 中 `x * y / gcd` 及 `CheckOverlapCoverage` 的 `SafeMultiplyShape`。检查点：大 shape 场景下确认溢出标志被处理。典型报错：负面积 / 覆盖性误判。
- **S017**：删除/替换 op 时 consumer 关系未同步。触发点：`RemoveOps` 先 `UpdateOperandBeforeRemoveOp` 再 `SetAsDeleted`，若 consumer list 未双向更新下游可能访问已删除 op。检查点：运行图不变量检查。典型报错：访问已删除 op / producer-consumer 不一致。

---

### 6. 常见被误判为 Pass 问题的症状

| 症状 | 实际常见根因模块 | 快速验证方法 |
|------|------------------|--------------|
| Assemble/view shape  mismatch 或 validshape 过大 | operation / interpreter（N001、N028、N044） | Dump `validshape`，对比 `ExecuteOpReshape`/`CheckViewValidShapesConstraint` 输出 |
| 拆分后 L0C/L1 memory type 错误 | `AssignMemoryType` / codegen（P008、N033） | 检查拆分前后 `GetMemoryTypeOriginal` 链，dump memory type |
| 动态 shape 下数据错位 | frontend / interpreter（C005、N016） | 确认前端 raw shape 与符号绑定；检查 `SymbolicScalar` 求值 |
| 运行期 AICore misalignment | codegen / operation（N002、N030） | 检查 `AssignMemoryType`/`CodegenPreproc` 后的 tile shape 与对齐 |

---

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`。

本词条补充检查项：

- [ ] 拆分后是否重新调用 `InferShapeUtils::InferShape`？新增 tensor 是否被加入 `addedOps_`？

---

### 8. 调试快速入口

- 开启 Pass dump：

  ```python
  import pypto
  pypto.set_pass_config("PVC2_OOO", "SplitLargeFanoutTensor",
                        pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
  ```

  或设置环境变量 `TILEFWK_CONFIG_PATH` 指向自定义 `tile_fwk_config.json`，开启 `pass.default_pass_configs.dump_graph`。
- 关键中间状态：`toInfoMap_`、`fromInfoMap_`、`lcmShapes`、`tileOffsets`、`overlaps`/`dualOverlaps`、`addedOps_`、`EraseRedundantAssembleOp` 删除列表。
- 推荐先跑的 UT：`SplitLargeFanoutTensorTest.*`（`framework/tests/ut/passes/src/test_split_large_fanout_tensor.cpp`），重点关注 `MtoM`、`1ToMGetCorrectAssemble`、`TestInferShapeHaveDynValidShape`。
- 相关 checker：`framework/src/passes/pass_check/merge_view_assemble_checker.cpp`（`MergeViewAssembleUtils::MergeViewAssemble` 调用）、`framework/src/passes/pass_check/split_reshape_checker.cpp`（相邻 Pass）。

---

---

## AssignMemoryType

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | AssignMemoryType |
| 所属目录 | `framework/src/passes/tile_graph_pass/data_path/` |
| 主要源文件 | `assign_memory_type.cpp`、`assign_memory_type.h`；辅助 `convert_op_inserter.h/cpp`、`memory_path_utils.h` |
| Pipeline 阶段 | tile_graph（data_path） |
| 前置依赖 Pass | SplitLargeFanoutTensor、DuplicateOp；依赖 InferTensorFormat / AutoCast / InferMemoryConflict 等已设定的 shape/format |
| 后置消费 Pass | InferDiscontinuousInput、RemoveRedundantOp、InsertOpForViewAssemble、ProcessAtomic、GraphPartition、GenerateMoveOp、CommonOperationEliminate 等 |
| 对应 bug 模式 | P008、P015、S001、S002、S003、S005、S006、S008、C001、C007 |

---

### 2. 设计目标

为每个 `LogicalTensor` 确定原始内存类型（`MemoryTypeOriginal`）以及每个 consumer 的需求内存类型（`toBeMap`），并在 view/assemble/reshape 等无法直连的场景中插入 `OP_CONVERT`，为后续 `GenerateMoveOp` 与 `OoOSchedule` 的 buffer 分配提供统一的内存类型约束。

---

### 3. 核心不变量

- 不变量 1：离开本 Pass 后，图中所有 tensor 的 `GetMemoryTypeOriginal()` 不得为 `MEM_UNKNOWN`（PostCheck `CheckTensorNotMemUnknown` 校验）。
- 不变量 2：`OP_VIEW` / `OP_VIEW_TYPE` / `OP_ASSEMBLE` 的属性（`ViewOpAttribute::to_`、`AssembleOpAttribute::from_`）必须与推断出的内存类型一致（`SyncViewMemoryAttr`、`SyncAssembleMemoryAttr`）。
- 不变量 3：任意 `MOVE_LOCAL` 类 op 的输入/输出内存类型必须存在平台直连路径（PostCheck `CheckMoveOpReachable`）。
- 不变量 4：`ConvertInserter::DoInsertion` 新增的 convert op 必须随后触发 `InferShapeUtils::InferShape`。
- 不变量 5：view 链上的内存类型推断必须沿前向视图链传播，不能仅看直接生产者（`InferTargetTypeThroughForwardViews`）。

---

### 4. 关键设计决策与取舍

| 决策 | 选择 | 原因 | 如果改这里要注意什么 |
|------|------|------|---------------------|
| 视图内存类型回退 | 当 `CanUseDirectViewPath` 返回 false 时强制输入回退到 `MEM_DEVICE_DDR` | 避免不支持的跨内存类型直连导致 codegen 失败 | 修改 `CanUseDirectViewPath` / `TryHandleSpecialDirectMemoryPath` 必须同步更新 `CheckMoveOpReachable` 与 `Platform::GetDie().HasDirectPath` 的判定 |
| 超大本地 buffer 回退 | 当 UB/L1 超过阈值（`UB_THRESHOLD_ASSEMBLE`、`UB_THRESHOLD_NORMAL`、`L1_THRESHOLD`）时强制回退 DDR | 防止 on-chip 内存超限 | 阈值是平台相关 magic number（S008），新增 memory type 时 `IsOversizedLocalBuffer` 必须显式处理 |
| 并行者需求冲突处理 | `HasParallelDifferentConsumerRequirement` 会阻断特殊直连路径 | 保证多 consumer 场景下各 consumer 都能拿到所需 dtype/memory type | 修改直连路径前必须检查是否引入 P015 所述的 parallel-consumer 错误 |

---

### 5. 已知脆弱场景

- **模式 P008**：视图链内存类型推断只看了直接 producer。触发条件：`OP_VIEW_TYPE` 前面还有多层 `OP_VIEW` 且直接 producer 为 `MEM_UNKNOWN`。检查点：`InferViewTypeMemoryType` 中 `InferTargetTypeThroughForwardViews` 是否被正确调用。典型报错：`Tensor[X]'s memoryType is still unknown`。
- **模式 P015**：强制 DDR/直连路径时忽略并行 consumer 的不同需求。触发条件：`CanUseDirectViewPath` / `TryHandleSpecialDirectMemoryPath` 对多 consumer tensor 开启直连。检查点：`HasParallelDifferentConsumerRequirement` 调用位置。典型报错：`CheckMoveOpReachable` 失败或运行时地址冲突。
- **模式 S001**：对 `GetOOperands()` / `GetIOperands()` 直接取 `.front()` 未判空。触发条件：异常图中 op 无输入/输出。检查点：`GetFirstInputOutputIfOpcode`、所有 `operation.iOperand.front()` 处。
- **模式 S003**：`std::dynamic_pointer_cast<ViewOpAttribute/AssembleOpAttribute>` 未判空。触发条件：op attribute 类型不匹配。检查点：`AssignViewAttrMemoryType`、`InferViewMemoryType`、`SyncAssembleMemoryAttr`。
- **模式 S006**：`CalcNZTensorSize`、`CalcLineOffset`、`ProcessL0C2UB...` 中 shape/stride 连乘可能溢出。触发条件：极大 shape 或动态符号。检查点：中间变量类型是否为 `int64_t`、是否做溢出检查。
- **模式 C001 / C007**：spill/同步/地址报错或 codegen tile 报错时，根因可能是本 Pass 推断的 memory type 与 dtype 组合不被下游支持。检查点：对比本 Pass 输出与 `codegen` 对 memory type/dtype 的支持矩阵。

---

### 6. 常见被误判为 Pass 问题的症状

| 症状 | 实际常见根因模块 | 快速验证方法 |
|------|------------------|--------------|
| `CheckMoveOpReachable` 报错某 MOVE_LOCAL 路径不可达 | `InferTensorFormat` / `AutoCast` 遗留了非法 format/dtype 组合，或 `Platform` 路径表未覆盖新内存类型 | 查看该 op 输入/输出 `MemoryTypeOriginal` 与 `Datatype`，确认 `Platform::Instance().GetDie().HasDirectPath` |
| 运行时 `rtMalloc` / workspace OOM | `SetHeuristicTileShapes` 产生过大 tile，或 `GlobalMemoryReUse` 未生效 | 对比 `workspaceOffset` 与本 Pass 中 DDR tensor 总大小 |
| OoO spill 失败 / local buffer 分配失败 | `SetHeuristicTileShapes` / `AddAlloc` 导致单 buffer 过大，或 `OoOScheduler` 双 dst 配置 | 开启 OoO `health_check`，检查 `localBufferMap_` 中最大 buffer size |
| codegen 报 unsupported memory type | `codegen` 对特定 dtype + memory type 组合不支持 | 查看 `AssignMemoryType` 为该 tensor 设置的类型是否命中 codegen 黑名单 |

---

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`。

---

### 8. 调试快速入口

- 开启 Pass dump 的命令/环境变量：在 `framework/src/interface/configs/tile_fwk_config.json` 中设置 `global.pass.default_pass_configs.dump_graph=true`、`print_graph=true`；或按策略为 `AssignMemoryType` 单独配置 `"dump_graph": true`。日志级别通过 `global.log_level` 调整。
- 关键中间状态：每个 tensor 的 `MemoryTypeOriginal`、`toBeMap`（consumer requirement）、`ViewOpAttribute::to_` / `AssembleOpAttribute::from_`、新增 convert op 列表。
- 推荐先跑的 UT：`framework/tests/ut/passes/src/test_assign_memory_type.cpp`。
- 相关 checker：`framework/src/passes/pass_check/assign_memory_type_checker.cpp`、`framework/src/passes/pass_check/assign_memory_type_checker.h`。

---

---

## ReduceCopyMerge

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | ReduceCopyMerge（源文件为 `reduce_copy.cpp`） |
| 所属目录 | `framework/src/passes/tile_graph_pass/graph_partition/` |
| 主要源文件 | `reduce_copy.cpp`、`reduce_copy.h` |
| Pipeline 阶段 | `PVC2_OOO` 策略，tile graph 阶段 |
| 前置依赖 Pass | `GraphPartition`、`NBufferMerge`、`L1CopyInReuseMerge` |
| 后置消费 Pass | `IntraSubgraphAdapter`、`GenerateMoveOp`、`CommonOperationEliminate` |
| 对应 bug 模式 | P013、P019、P021、S001、S002、S004、S005、S006、S007、S008、S014、S015 |

---

### 2. 设计目标

在 DAV_3510 上合并被边界 tensor 连接的细粒度 subgraph，以减少跨 subgraph 的 copy；同时避免引入环、违反 latency/AIV-AIC 比例约束，以及防止合并后出现“内部 tensor 仍有外部使用”的情况。

---

### 3. 核心不变量

- 仅在 `NPUArch::DAV_3510` 上启用。
- 每个 op 的 subgraph ID 必须通过 `op.UpdateSubgraphID` 一致更新。
- 合并不能引入环（`CanMergeWithoutCycle` / `HasCycle`）。
- 合并后的 subgraph 必须同时包含 AIC 和 AIV 算子，且总 latency 与 AIV/AIC 比例在限定范围内。
- 被合并为内部 tensor 的边界 tensor 不能仍有外部 endpoint（`CheckNoExternalUseOfMergedInnerTensor`）。
- `subgraphIdUpdated` 必须是紧凑的合法映射。

---

### 4. 关键设计决策与取舍

| 决策 | 选择 | 原因 | 如果改这里要注意什么 |
|------|------|------|---------------------|
| 合并候选生成 | 按边界 tensor 分组及输出/输入 tensor size 排序 | 优先消除高 copy 代价边界 | 排序/迭代顺序变化会影响确定性 |
| 防环 | Union-Find + 合并后图拓扑检查 | 跨依赖边合并会导致死锁 | 必须同时更新 inGraph/outGraph |
| Latency/比例保护 | `maxLatency=1e7`、`aivRatio={1e-6,1e6}` | 防止融合反而降性能 | 魔数需平台化或配置化 |
| 强制合并 | `IsEnforceMergeBoundary` 在 CV fuse scope ID 一致时触发 | CV fusion 要求同 scope | 过度强制会破坏 scope 同步 |

---

### 5. 已知脆弱场景

- **P013**：hash/顺序敏感。触发点：`UpdateMergeInput` 用 `std::set<std::vector<int>> visitedMergeGroup` 去重，虽然 vector 已排序，但 `sortedMergeGroup` 来自 `std::multimap`，迭代顺序依赖 key。检查点：同测试多次运行比较 `subgraphIdUpdated`。典型报错：非确定性 subgraph 编号。
- **S004**：视图类 OP 处理不完整。触发点：`MarkCrossSubgraph` 只检查 `OP_RESHAPE`，忽略 `OP_VIEW`/`OP_ASSEMBLE` 跨 subgraph。检查点：构造 view/assemble 跨 subgraph 的图。典型报错：误合并导致数据依赖断裂。
- **S005**：内存类型推断边界过窄。触发点：`MarkNoMergeSubgraph` 仅检查 `MEM_DEVICE_DDR` 的内部 tensor。检查点：边界 tensor 的 L0C/L1 路径。典型报错：copy 方向错误 / 非法内存访问。
- **S006**：整数溢出。触发点：`subgraphAICLatency[src] += opLatency`、`subgraphInputSize`/`subgraphOutputSize` 累加。检查点：超大 latency 或 tensor size。典型报错：负数 latency / 内存分配异常。
- **S008**：硬编码常量。触发点：`maxLatency`、`aivRatio`、`betweenSubgraphScheduleTime=1500` 等字面量。检查点：提取到 platform/arch 配置。典型报错：不同平台行为不一致。
- **S014**：指针有序容器导致非确定性。触发点：`std::unordered_set<int> noMergeSubgraph` 等被遍历使用时顺序不稳定。检查点：使用稳定 ID 排序。典型报错：偶发结果不一致。

---

### 6. 常见被误判为 Pass 问题的症状

| 症状 | 实际常见根因模块 | 快速验证方法 |
|------|------------------|--------------|
| 合并后仍有跨 subgraph copy | `GraphPartition` / `NBufferMerge`（初始子图划分） | Dump 合并前 subgraph ID，检查 `GraphPartition` 边界代价 |
| 合并后性能回退 | codegen / machine 调度 | 对比 `EstimateExecTime` 估计值与 NPU 实测 profiling |
| subgraph ID 越界或缺失 | `PassManager` 状态污染（S027） | 确认 `PassManager::startIdx` 未跨策略污染 |
| 合并后检测到环 | 下游 `InsertSync` / `OoOSchedule` 或原图问题 | 在 `OoOSchedule` 后跑环检测 |

---

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`。

本词条补充检查项：

- [ ] 修改 latency/size 累加时是否使用 int64_t / 饱和加法？

---

### 8. 调试快速入口

- 开启 Pass dump：

  ```python
  pypto.set_pass_config("PVC2_OOO", "ReduceCopyMerge",
                        pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
  ```

- 关键中间状态：`mergeInput.mergeGroup`、`noMergeSubgraph`、`noMergeSubgraphEnforce`、`boundaryTensors`、`subgraphIdUpdated`、`EstimateExecTime` 结果。
- 推荐先跑的 UT：`ReduceCopyTest.*`（`framework/tests/ut/passes/src/test_reduce_copy.cpp`），重点关注 `PreserveOriginalSubgraphId`、`TestCase3`。
- 相关 checker：无专属 checker；可配合 `framework/src/passes/pass_check/pre_graph_checker.cpp`、`framework/src/passes/pass_check/intra_subgraph_adapter_checker.cpp` 使用。

---

---

## SupernodeGraphBuilder

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | SupernodeGraphBuilder |
| 所属目录 | `framework/src/passes/tile_graph_pass/graph_partition/` |
| 主要源文件 | `supernode_graph_builder.cpp`、`supernode_graph_builder.h`；作为 `IsoPartitioner` 的基类被 `graph_partition.cpp` 调用 |
| Pipeline 阶段 | tile_graph（`GraphPartition` Pass 内部，用于 Iso 分区） |
| 前置依赖 Pass | `AssignMemoryType`（memory type 已定）、`ProcessAtomic`、各类 split/duplicate/reshape 优化 Pass |
| 后置消费 Pass | `IsoPartitioner::BuildIsomorphismGroups`、`IsomorphismGroupMergeProcess`、`UpdatePartitionResult`，后续 `NBufferMerge`、`L1CopyInReuseMerge`、`GenerateMoveOp`、`CommonOperationEliminate` |
| 对应 bug 模式 | P007、S004、S007、S012、S014、C002 |

---

### 2. 设计目标

把 tile 子图按数据依赖、core 类型、copy 方向、scope 等约束合并成 supernode，为 `IsoPartitioner` 提供同构子图识别与并行划分的粒度，最终写入 `subgraph_id`。

---

### 3. 核心不变量

- 不变量 1：合并后的 supernode 内部不能有环（`NodeGraphInfo::AvoidLoop`）。
- 不变量 2：合并节点的 core 类型必须满足 `OperationGraphInfo::CoreTypeMergeable`（最多两种非 AICPU/HUB 类型，或 CVMix 场景下单一类型）。
- 不变量 3：`OP_VIEW` / `OP_ASSEMBLE` 的合并必须满足 `CheckUbToUbWithDynOffset`、`CheckViewAssembleOffset`、`CheckScopeNotMergeable`。
- 不变量 4：每个 op 必须被分配到唯一 node（`op2Node_`），每个 node 的 op 列表非空。
- 不变量 5：scope 内若同时出现 cube 与 vector，则仅在 `IsCVMixPlatform()` 时允许，否则必须拆分。

---

### 4. 关键设计决策与取舍

| 决策 | 选择 | 原因 | 如果改这里要注意什么 |
|------|------|------|---------------------|
| 多维度合并规则 | `ConvertCombine`、`L1CopyInCombine`、`AssembleCombine`、`CopyOutCombine`、`CopyInCombine`、`MulAccCombine` 分别处理不同场景 | 把必须同子图的 op 强制绑定 | 新增 op code 或 copy 方向时必须补充对应 Combine 规则 |
| 基于 hash 的同构识别 | `BuildHashValues` 计算 node 前向/后向 hash，用 `hash2NodeMap_` 聚类 | 快速找出可并行的同构子图 | 修改 `CombineHash` 或 `GetHash` 必须保证顺序无关与稳定性（S007） |
| Scope 合并二次处理 | `ProcessScopeMerge` 在 supernode 构建后再按 scope 合并 | 满足前端/融合的 scope 约束 | 修改 `allowParallelMerge` 逻辑必须同步 `ValidateScopeCoreTypes` |

---

### 5. 已知脆弱场景

- **模式 P007**：reshape/assemble/view 合并条件未覆盖多输入输出、动态 shape、axis-combine padding。触发条件：`GetNodeMergeable` 对特殊 reshape 节点放行。检查点：`nodeInGraph_` / `nodeOutGraph_` 大小判断、`CheckUbToUbWithDynOffset`、`CheckViewAssembleOffset`。典型报错：子图内出现非法 reshape 或 padding 轴。
- **模式 S004**：视图类 op 处理不完整。触发条件：只处理 `OP_VIEW` 与 `OP_ASSEMBLE`，忽略 `OP_RESHAPE` 或 `OP_VIEW_TYPE`。检查点：所有 `GetOpcode() == Opcode::OP_VIEW` 分支是否同时覆盖 `OP_RESHAPE` / `OP_VIEW_TYPE`。
- **模式 S007 / S014**：hash 计算依赖顺序或容器 key 为指针。触发条件：`OperationGraphInfo::GetHash` 拼接字符串、node 内 op 顺序不一致。检查点：`ComputeDirectionalNodeHash` 中对 neighborHashes 排序、`CombineHash` 是否满足交换律。
- **模式 S012**：边界检查不一致。触发条件：`L1CopyInCombine`、`ConvertCombine` 等函数中 `i` 的范围判断不一致（`i < opList.size()` 与 `i <= opList.size()`）。检查点：所有 helper 的边界条件。
- **模式 C002**：合并后 shape / validshape 错误。触发条件：上游 `SplitLargeFanoutTensor` / `RemoveRedundantReshape` 改变了 view 链。检查点：dump 合并前后的 shape 与 `validshape`。

---

### 6. 常见被误判为 Pass 问题的症状

| 症状 | 实际常见根因模块 | 快速验证方法 |
|------|------------------|--------------|
| 子图划分过大或过小 | `GraphPartition` 参数 `sgParallelNum` / `sgPgLowerBound` 配置，或 `EstimateCycleUB` 的 latency 阈值 | 查看 `cycleUB_`、`parallelNum_` 与 `IsoPartitioner::SuitableForMergeCheck` 日志 |
| 同构子图未被识别 | `BuildHashValues` 输入中包含了 memory type 或 op 顺序差异 | 对比两个预期同构子图的 `nodeHashList_` 与 `opHashList_` |
| 子图边界出现非法 COPY_IN / COPY_OUT | `PreGraphProcess` / `GenerateMoveOp` 边界处理错误 | 查看 `GraphPartitionChecker::PostOperationCheck` 输出 |
| cube/vector 混到一个子图报错 | 非 CVMix 平台下 `ProcessScopeMerge` 未拆分 | 检查 `GraphUtils::IsCVMixPlatform()` 与 scope 内 `hasCube`/`hasVector` |

---

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`。

---

### 8. 调试快速入口

- 开启 Pass dump 的命令/环境变量：`tile_fwk_config.json` 中 `GraphPartition`（identifier 为 `GraphPartition`）配置 `"dump_graph": true`、`"print_graph": true`。
- 关键中间状态：`operationInfo_->opHashList_`、`superNodeInfo_->nodeHashList_`、`hash2NodeMap_`、`node2Op_`、`nodeInGraph_` / `nodeOutGraph_`、`nodeMergeable_`、`scopeId` 分布。
- 推荐先跑的 UT：`framework/tests/ut/passes/src/test_graph_partition.cpp`。
- 相关 checker：`framework/src/passes/pass_check/iso_partitioner_checker.cpp`、`framework/src/passes/pass_check/iso_partitioner_checker.h`。

---

---

## OspPartitioner

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | OspPartitioner |
| 所属目录 | `framework/src/passes/tile_graph_pass/graph_partition/` |
| 主要源文件 | `osp_partitioner.cpp`、`osp_partitioner.h` |
| Pipeline 阶段 | tile_graph |
| 前置依赖 Pass | `GraphPartition`（构建 `OperationInfo` / `SuperNodeInfo`）、`AssignMemoryType`、`InsertOpForViewAssemble`、`ProcessAtomic` |
| 后置消费 Pass | `SubgraphToFunction`（按 `subgraph_id` 生成 leaf function）、`IntraSubgraphAdapter`、`GenerateMoveOp` |
| 对应 bug 模式 | S018、S019、S020、S021、S022、S023、S024、S025、S028、S029 |

### 2. 设计目标

将 tile-graph 上的 super-node DAG 进一步划分为若干子图（subgraph），使得每个子图满足 core-type 兼容性、内存上限与通信代价约束；输出为每个 operation 的 `subgraph_id`，供 `SubgraphToFunction` 生成 leaf function。

### 3. 核心不变量

- 每个 super-node 必须被分配到且仅到一个 `subgraph_id`。
- `vertexContractionMap.size()` 必须与 `superNodeInfo_->node2Op_.size()` 保持一致；否则 `UpdatePartitionResult` 会越界或漏更新。
- CV-Mix 模式下 `numVectorCores % numCubeCores == 0`，否则架构构建直接失败。
- `BuildHashValues` 在 `numOps == 0` 时必须安全返回，不能出现 `int32_t` 下溢。
- 同构合并（MERKLEBSP）必须保证 Merkle hash 碰撞不会导致不同构子图被错误合并。

### 4. 关键设计决策与取舍

| 决策 | 选择 | 原因 | 如果改这里要注意什么 |
|------|------|------|---------------------|
| 分区算法 | 支持 `SARKAR` 与 `MERKLEBSP` 两种模式 | Sarkar 适合通用 DAG 粗化；MERKLEBSP 利用 Merkle hash 发现同构子图以提升并行度 | 新增模式时必须补充 `ConstructBspArch*/ConstructDag*` 分支，并更新 `iso_partitioner_checker.cpp` |
| CV-Mix vs CV-Split | 由 `useCVMixPartition_` 控制 | Mix 模式把 cube + 对应数量 vector 绑定为一个混合处理器；Split 模式独立处理 | 修改 `cubeVecMemoryBound` 计算时要防止 `static_cast<WorkType>` 窄化溢出（S021） |
| 同构判定 | Merkle hash 作为必要条件，后续由 `IsomorphicSubgraphScheduler` 调度 | 快速过滤非候选；但存在理论碰撞风险 | 不能把 Merkle hash 当作充分条件，需加 canonical-label 二次校验或冲突检测（S022） |
| SuperNode 权重 | 通信权重按跨子图 consumer 的 `MemorySize()` 累加 | 反映真实 DMA/UB 搬运量 | 累加类型必须能容纳大 tensor，避免 `WorkType` 回绕 |

### 5. 已知脆弱场景

- **模式 S018**：`DetermineEffectiveMinProcCount` 中 `typeCount.empty()` 分支赋值后未 `return`，随后执行 `*std::min_element(typeCount.begin(), typeCount.end())` 导致空向量解引用。触发条件：MERKLEBSP 同构调度遇到零处理器类型。检查点：`framework/src/passes/algorithms/osp/dag_divider/isomorphism_divider/isomorphic_subgraph_scheduler.h`。典型报错：`SIGSEGV` 在 `min_element`。
- **模式 S019**：`lazy_communication_cost.h` 中只检查 `stepNeeded[proc] < numberOfSupersteps`，未校验 `>= staleness`，直接 `stepNeeded[proc] - staleness` 下溢。触发条件：通信延迟大于实际步数。检查点：`lazy_communication_cost.h` 中 send 代价计算。典型报错：`size_t` 下溢导致异常大的通信代价。
- **模式 S020**：`DagVectorAdapter::SetInOutNeighbors` 存储调用者 vector 的裸指针；`superNodeInfo_` 重建后 OSP 图 dangling。触发条件：分区前多次重建 super-node 图。检查点：`ConstructDagCVMix` 后是否重置 `superNodeInfo_`。
- **模式 S021**：`memorySize` / `cubeVecMemoryBound` 被 `static_cast<WorkType>(int32_t)` 截断，大 tensor 权重变负。触发条件：L1+UB 总和超过 INT32_MAX。检查点：`ConstructBspArchCVMix` 中 `procMemoryBound[i]` 赋值。
- **模式 S022**：`AreIsomorphicByMerkleHash` 仅必要条件，但 `IsMergeViable` 把它当唯一同构判定。触发条件：结构不同但 Merkle hash 相同的子图。检查点：`merkle_hash_computer.h`。典型报错：后续 `CODEGEN_PREPROC` 算子缺失或调度异常。
- **模式 S023**：`std::accumulate(..., 0U)` 求和结果超过 `UINT_MAX` 回绕。触发条件：平台配置的处理器类型数量之和极大。检查点：`bsp_architecture.h`。
- **模式 S024**：`SetDiagonalCompatibilityMatrix` 不校验 `numberOfTypes>0`；`IsCompatibleType` 直接下标访问。触发条件：异常架构配置或 DAG 类型数与架构类型数不一致。
- **模式 S025**：`ApplyMove` 直接用 `move.toStep_/toProc_` 索引多维数组，未校验边界。触发条件：KL 局部搜索收到异常初始解。检查点：`kl_active_schedule.h`。
- **模式 S028**：`BuildHashValues` 反向循环 `for (int32_t i = static_cast<int32_t>(numOps - 1); i >= 0; i--)` 在空图时 `numOps-1` 下溢。触发条件：空函数或全合并后的图触发 MERKLEBSP。检查点：`osp_partitioner.cpp:493`。
- **模式 S029**：`UpdatePartitionResult` 假设 `vertexContractionMap.size() == superNodeInfo_->node2Op_.size()`，coarsening 输出不同步时越界。触发条件：Sarkar/MerkleBsp 返回的收缩映射与超点集合不一致。检查点：`UpdatePartitionResult` 中的双重 size 校验。

### 6. 常见被误判为 Pass 问题的症状

| 症状 | 实际常见根因模块 | 快速验证方法 |
|------|------------------|--------------|
| 子图数量异常少 / 所有 op 被分到同一 subgraph | `GraphPartition` 前置 super-node 合并过度，或 `paramConfigs_.sgPgLowerBound` 配置过大 | 检查 `superNodeInfo_->node2Op_` 大小；对比 `partitionWorkLowerBound_` |
| 后续 `SubgraphToFunction` 报 subgraph_id 越界 | `UpdatePartitionResult` 的 `vertexContractionMap` 与 `node2Op_` 不一致（S029），或上游 `GraphPartition` 未正确初始化 | dump `vertexContractionMap` 与 `node2Op_.size()` |
| 编译挂死在 MERKLEBSP | `IsomorphicSubgraphScheduler` / `merkle_hash_computer.h` 哈希碰撞或图 dangling（S020/S022） | 开启 pass dump，检查 hash 冲突 |
| 分区后某个子图内存超过 L1/UB | `AssignMemoryType` 对 tensor memory type 推断错误，或 `SetVertexCommMemWeight` 权重截断（S021） | 对比 `superNodeInfo_->nodeCycles_` 与 `MemorySize()` |
| 同构子图未合并导致性能差 | 不是 bug，可能是 `useCVMixPartition_` 开关或 `kIsoSchedulerWorkThreshold` 阈值导致 | 检查 OSP mode 与 threshold 配置 |

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`。

---

### 8. 调试快速入口

- 开启 Pass dump 的命令/环境变量：`CFG_COMPILE_DEBUG_MODE=CFG_DEBUG_ALL`（开启所有 pass 的 printGraph/dumpGraph）；或单独配置 `pass_configs.json` 中 `OspPartitioner` 的 `dumpGraph=true`
- 关键中间状态：`superNodeInfo_->node2Op_`、`nodeOutGraphList_`、`nodeHashList_`、`vertexContractionMap`、`operationInfo_->opHashList_`
- 推荐先跑的 UT：`framework/tests/.../tile_graph_pass/*osp*`（若存在）；否则跑包含 `GRAPH_PARTITION` 的端到端 case
- 相关 checker：`framework/src/passes/pass_check/iso_partitioner_checker.cpp`

---

## TileAssignMemoryType

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | TileAssignMemoryType |
| 所属目录 | `framework/src/passes/tile_graph_pass/data_path/` |
| 主要源文件 | `assign_memory_type.cpp`、`assign_memory_type.h` |
| Pipeline 阶段 | tile_graph |
| 前置依赖 Pass | `RemoveRedundantOp`、`InsertOpForViewAssemble`、`ProcessAtomic`、`GraphPartition` |
| 后置消费 Pass | `GenerateMoveOp`、`ConvertOpInserter`、`IntraSubgraphAdapter`、`SubgraphToFunction` |
| 对应 bug 模式 | S001、S002、S003、S005、S006、S008、S009、P008、P015 |

### 2. 设计目标

为 tile-graph 中的每个 tensor 推断并固定其原始内存类型（original）与消费者需求内存类型（requirement），在冲突处插入 `OP_CONVERT` 或 move op，保证后续 codegen 的 memory path 合法。

### 3. 核心不变量

- 每个 tensor 的 `MemoryTypeOriginal()` 与所有 consumer 的 `MemoryTypeRequirement()` 必须能够匹配；否则必须通过 `InsertConvertOpsAndInferShape` 插入转换 op。
- `OP_VIEW` 的输出原始类型必须与其 `ViewOpAttribute::GetTo()` 一致；`OP_ASSEMBLE` 的输入需求类型必须与其 `AssembleOpAttribute::GetFrom()` 一致。
- `AssignMatmulInputRequirements` 中 matmul 输入若来自另一个 matmul，则需求为 `MEM_L0C`；若来自 `OP_VIEW`，则按 view attr 推导。
- `SyncViewAssembleMemoryAttrs` 必须保证 view/assemble 链上内存类型一致，不能出现链中间断。
- 动态 shape / validshape 必须在创建新 tensor 时传递，不能丢失。

### 4. 关键设计决策与取舍

| 决策 | 选择 | 原因 | 如果改这里要注意什么 |
|------|------|------|---------------------|
| 三阶段推断 | `AssignConfirmedMemoryTypes` -> `InferUncertainMemoryTypes` -> `ResolveMemoryUnknowns` | 先处理 opcode/view/assemble 的强制约束，再传播未知，最后兜底 | 新增特殊 op 时必须同时更新三阶段，否则会出现未知类型残留 |
| L0C2UB pattern | 单独识别 `OP_RESHAPE` 介于 cube-assemble 与 vector-view 之间的链 | 允许 L0C 数据通过 reshape 直接到 UB，避免回 DDR | 修改 `IsReshapeCubeToVecL0C2UBPattern` 时要保证 producer/consumer 判空 |
| View 链内存类型 | `AssignViewAttrMemoryType` 直接从 view attr 取 `GetTo()` | 避免从直接 producer 推断导致链上类型错误 | 若支持嵌套 view，需要向前追踪到链根（P008） |
| 冲突解决 | `ConvertOpInserter` 统一插入 convert | 集中管理 dtype / memory type 转换 | 修改转换条件时要同步更新 `memory_path_utils.cpp` |

### 5. 已知脆弱场景

- **模式 S001**：直接解引用 `GetProducers()` / `GetConsumers()` 的 `.front()` / `.begin()` / `[0]` 未判空。触发条件：孤立 tensor 或异常图结构。检查点：`AssignMatmulInputRequirements`、`IsReshapeCubeToVecL0C2UBPattern` 中的循环。
- **模式 S002**：多 producer/consumer 只取第一个。触发条件：一个 tensor 有多个 producer。检查点：matmul 输入推导中的 `for (const auto& producerOp : tensor->GetProducers())` 是否正确聚合冲突需求。
- **模式 S003**：`dynamic_pointer_cast<ViewOpAttribute>` / `AssembleOpAttribute` 后未判空。触发条件：op attribute 类型不匹配。检查点：`AssignViewAttrMemoryType`、`AssignAssembleAttrMemoryType`。
- **模式 S005**：内存类型推断边界过窄，只处理 DDR/L0C/L1/UB 中的部分组合。触发条件：新 opcode 或新 memory type。检查点：`OpcodeManager::Inst().GetInputsMemType` 的枚举。
- **模式 S006**：shape/stride 相乘或 `MemorySize()` 累加时 int64_t 溢出。触发条件：大 shape。检查点：权重与 buffer size 计算。
- **模式 S008**：硬编码阈值或平台常量。触发条件：跨平台迁移。检查点：代码中除 0/1/-1 外的数字字面量。
- **模式 S009**：动态 shape / validshape 处理不当。触发条件：view/assemble 链含动态维。检查点：`InsertConvertOpsAndInferShape` 创建新 tensor 时是否传递 `DynValidShape`。
- **模式 P008**：view tensor 的内存类型从直接 producer 推断，而不是沿 forward view chain 找到原始 buffer。触发条件：嵌套 view。检查点：view 链内存传播逻辑。
- **模式 P015**：为特殊 case 强制回退到 DDR 时忽略并行 consumer 或 OoO 约束。触发条件：并行 consumer 要求不同 memory type。检查点：`ResolveMemoryUnknowns` 中强制 DDR 的分支。

### 6. 常见被误判为 Pass 问题的症状

| 症状 | 实际常见根因模块 | 快速验证方法 |
|------|------------------|--------------|
| 运行时报 `MTE2` / `fixpipe` 路径非法 | `machine` 的 launch/runtime 对 memory type 组合支持有限 | 检查 pass 输出的 memory type 与 `machine` 支持矩阵 |
| `OP_CONVERT` 过多导致性能差 | 上游 `RemoveRedundantOp` / `MergeViewAssemble` 未合并可消除的 view/assemble | dump tile-graph 前后对比 convert 数量 |
| matmul 输入精度异常 | `operation` 中 matmul 的 format/dtype 约束与 pass 推断的 memory type 不匹配 | 检查 matmul op 的 input memory type 支持列表 |
| 动态 shape 编译失败 | `InferDynShape` / `PreGraph` 未正确传递 `DynValidShape`，导致本 pass 创建静态 tensor | dump validshape 链 |
| `CodegenPreproc` 报 buffer 未对齐 | `PadLocalBuffer` / `AxisCombine` 未正确对齐，而非 memory type 错误 | 检查 alignment 属性 |

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`。

---

### 8. 调试快速入口

- 开启 Pass dump 的命令/环境变量：`CFG_COMPILE_DEBUG_MODE=CFG_DEBUG_ALL`；或在 `pass_configs.json` 中启用 `AssignMemoryType` 的 `dumpGraph`/`printGraph`
- 关键中间状态：每个 tensor 的 `MemoryTypeOriginal()`、`MemoryTypeRequirement()`、`viewOpAttribute->GetTo()`、`assembleOpAttribute->GetFrom()`
- 推荐先跑的 UT：`framework/tests/.../tile_graph_pass/assign_memory_type*`；`AssignMemoryTypeChecker` 相关单测
- 相关 checker：`framework/src/passes/pass_check/assign_memory_type_checker.cpp`

---

## InsertOpForViewAssemble

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | InsertOpForViewAssemble |
| 所属目录 | `framework/src/passes/tile_graph_pass/graph_optimization/` |
| 主要源文件 | `insert_op_for_viewassemble.cpp`、`insert_op_for_viewassemble.h` |
| Pipeline 阶段 | tile_graph |
| 前置依赖 Pass | `AssignMemoryType`（已确定 memory type） |
| 后置消费 Pass | `GenerateMoveOp`、`ConvertOpInserter`、`SubgraphToFunction` |
| 对应 bug 模式 | S001、S002、S003、S004、S009、S017、P009、P016 |

### 2. 设计目标

当 view 的输出直接作为 assemble 的输入但二者在 offset、shape、内存类型或动态维度上不匹配时，在两者之间插入 `OP_ASSEMBLE` + `OP_VIEW` 的 copy 对（DDR 中转），保证 assemble 消费到合法 tensor。

### 3. 核心不变量

- 插入 copy 后，原 `viewOp->oOperand` 与 `assembleOp->iOperand` 之间的数据流必须通过 `{ddrTensor}` 中转，不能破坏原有 producer/consumer 关系。
- 新 tensor 的 `DynValidShape` 必须与原 tensor 一致。
- 若原 assemble 输入已是 `MEM_DEVICE_DDR`，则直接将其原始类型与 view/assemble attr 改为 `MEM_UB`，不能重复插入 copy。
- `NeedInsertCopy` 判定必须覆盖 offset、dynOffset、shape、memory type 四维差异。

### 4. 关键设计决策与取舍

| 决策 | 选择 | 原因 | 如果改这里要注意什么 |
|------|------|------|---------------------|
| copy 对形式 | `assemble(viewOut -> ddrTensor)` + `view(ddrTensor -> moveInTensor)` | 利用 assemble/view 的 attr 描述 offset 与 validshape | 必须保证 `moveInTensor` 的 memory type 与 assemble 期望一致 |
| DDR 短路 | 若输入已是 DDR，则改 attr 为 UB 而不插 op | 减少冗余拷贝 | 修改 attr 后需同步 `SetMemoryTypeBoth` |
| 多 producer 处理 | `NeedInsertCopy` 遍历 `assembleOut->GetProducers()` | assemble 可能有多个输入 producer | `recordOpPair_` 必须记录所有需要处理的 view/assemble 对 |
| 图更新 | 使用 `assembleOp->ReplaceInput(moveInTensorPtr, moveOutTensorPtr)` | 统一更新 consumer 关系 | 必须确认 `ReplaceInput` 同时更新了双向边（S017） |

### 5. 已知脆弱场景

- **模式 S001**：`NeedInsertCopy` 中 `assOp->GetIOperands()[0]->GetProducers().begin()` 未判空。触发条件：assemble 输入 tensor 无 producer。检查点：`insert_op_for_viewassemble.cpp:77`。
- **模式 S002**：多 producer 场景只取第一个。触发条件：一个 tensor 有多个 producer。检查点：`NeedInsertCopy` 中对 `GetProducers().begin()` 的解引用。
- **模式 S003**：`std::dynamic_pointer_cast<ViewOpAttribute>` / `AssembleOpAttribute` 未判空；`std::static_pointer_cast` 也未判空。触发条件：op attribute 类型不匹配。检查点：`NeedInsertCopy` 中的 cast。
- **模式 S004**：视图类 OP 处理不完整，可能遗漏 `OP_VIEW_TYPE` 或 `OP_RESHAPE`。触发条件：链中出现非 `OP_VIEW` 的视图 op。检查点：`NeedInsertCopy` 中 `prodOp->GetOpcode() != Opcode::OP_VIEW` 分支。
- **模式 S009**：动态 shape / validshape 未传递。触发条件：插入 copy 时新 tensor 的 `DynValidShape` 丢失。检查点：`InsertViewAssemble` 中创建 tensor 时是否使用 `moveOutTensorPtr->GetDynValidShape()`。
- **模式 S017**：删除/替换 op 时 consumer 关系未同步。触发条件：`ReplaceInput` 未正确更新 producer/consumer 双向关系。检查点：插入 copy 后 `function.Operations()` 的 consumer 列表。
- **模式 P009**：克隆属性不完整。触发条件：插入的 view/assemble 未完整拷贝原 op 的所有属性。检查点：新 op 的 attribute 是否包含 offset、dynOffset、validshape。

### 6. 常见被误判为 Pass 问题的症状

| 症状 | 实际常见根因模块 | 快速验证方法 |
|------|------------------|--------------|
| assemble 输入报 memory type 不匹配 | 上游 `AssignMemoryType` 对 assemble 输入的 requirement 推断错误 | 检查 `AssignMemoryType` 输出的 memtype |
| 插入 copy 后 shape 推导失败 | `InferShapeUtils::InferShape` 或 `InferDynShape` 未处理新 view/assemble | 在 `RemoveRedundantOp` 后 dump 新 op 的 shape |
| 动态 shape 运行结果错 | `operation` 中 view/assemble 的 dynamic offset 实现与 pass 设置不一致 | 检查 runtime 中 dynOffset 是否被正确消费 |
| 出现大量冗余 copy | 不是 bug，可能是 `NeedInsertCopy` 阈值过严；或上游 view/assemble 未合并 | 统计 copy 对数量与原始 view/assemble 数量比 |
| consumer 关系断裂 | `ReplaceInput` 实现问题或本 pass 未调用图校验 | 开启 `KEY_ENABLE_PASS_VERIFY` |

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`。

---

### 8. 调试快速入口

- 开启 Pass dump 的命令/环境变量：`CFG_COMPILE_DEBUG_MODE=CFG_DEBUG_ALL`；或配置 `InsertOpForViewAssemble` 的 `dumpGraph=true`
- 关键中间状态：`recordOpPair_`、`assembleOutSet_`、`notProcessOut_`、每个 assemble 的 offset/dynOffset/shape/memtype
- 推荐先跑的 UT：`framework/tests/.../tile_graph_pass/insert_op_for_viewassemble*`；含 view-assemble 链的端到端 case
- 相关 checker：`framework/src/passes/pass_check/assemble_checker.cpp`、`framework/src/passes/pass_check/remove_redundant_op_checker.cpp`

---

## TileRemoveRedundantOp

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | TileRemoveRedundantOp |
| 所属目录 | `framework/src/passes/tile_graph_pass/graph_optimization/` |
| 主要源文件 | `remove_redundant_op.cpp`、`remove_redundant_op.h` |
| Pipeline 阶段 | tile_graph |
| 前置依赖 Pass | `GraphPartition`、`AssignMemoryType`、`InsertOpForViewAssemble` |
| 后置消费 Pass | `GenerateMoveOp`、`ConvertOpInserter`、`SubgraphToFunction` |
| 对应 bug 模式 | S001、S002、S004、S009、S012、S017、S045、P004、P007、P010、P019 |

### 2. 设计目标

删除 tile-graph 中输入输出在 shape、offset、memory type、动态 validshape 上完全等价的冗余 op（如冗余 reshape/view/assemble），并通过 `MergeViewAssembleUtils::MergeViewAssemble` 合并可折叠的 view-assemble 对，降低后续 codegen 的复杂度。

### 3. 核心不变量

- 删除或合并 op 后，所有剩余 consumer 的输入必须仍然能够正确解析到原数据。
- `ProcessRedundantOpWithoutDynShape` 中 `OP_ASSEMBLE` 多 producer 或并行 assemble + reshape consumer 时不能删除。
- 动态 shape 场景必须通过 `EqualInOut` 比较 `DynValidShape`。
- 合并 view-assemble 时必须禁止多输入/多输出、动态 shape、axis-combine pad 轴等场景（P007）。
- 调用 `DeadOperationEliminator::EliminateDeadOperation` 前必须先通过 `function.UpdateOperandBeforeRemoveOp` 更新 consumer 关系。

### 4. 关键设计决策与取舍

| 决策 | 选择 | 原因 | 如果改这里要注意什么 |
|------|------|------|---------------------|
| 分 opcode 集合处理 | `matchOpcodeWithDynshape` 与 `matchOpcodeWithoutDynshape` 两套集合 | 不同 op 对动态 shape 的敏感度不同 | 新增可删除 opcode 时必须明确归入哪一套 |
| assemble 特殊保护 | 并行 assemble + reshape consumer 不删除 | 防止删除后破坏 reshape 的输入格式 | 修改条件时要验证多 consumer 场景 |
| 迭代删除 | `while (operationUpdated)` 循环直到收敛 | 一次删除可能暴露新的冗余 op | 必须设置迭代上限或防止无限循环 |
| 合并委托 | 调用 `MergeViewAssembleUtils::MergeViewAssemble` | 复用统一合并逻辑 | 合并失败时会回退，需检查返回状态 |

### 5. 已知脆弱场景

- **模式 S001**：`ProcessViewAssemble`、`ProcessPerfectMatch`、`GenerateNewView` 中直接解引用 `op->iOperand.front()`、`op->oOperand.front()` 未判空。触发条件：异常 op 无输入/输出。
- **模式 S002**：`ProcessViewAssemble` 中 `auto& prodOp = *op.GetIOperands()[0]->GetProducers().begin()` 只取第一个 producer。触发条件：assemble 输入有多个 producer。
- **模式 S004**：视图类 OP 处理不完整，可能遗漏 `OP_VIEW_TYPE` 或 `OP_RESHAPE`。触发条件：链中出现非 `OP_VIEW` 视图 op。检查点：`ProcessViewAssemble` 中对 `OP_VIEW` 的特判。
- **模式 S009**：动态 shape / validshape 处理不当。触发条件：删除 op 时未比较 `DynValidShape`，或合并后未传递。检查点：`EqualInOut`、`MergeViewAssembleUtils`。
- **模式 S012**：原子操作/同步条件错误。触发条件：本 pass 删除或合并 op 后影响 `ProcessAtomic` 插入的原子标记。检查点：删除前后 `atomic` 相关 attr 是否一致。
- **模式 S017**：删除/替换 op 时 consumer 关系未同步。触发条件：`function.UpdateOperandBeforeRemoveOp` 未正确更新 producer/consumer。检查点：`RemoveDummyOp` 与 `DeadOperationEliminator` 调用顺序。
- **模式 P004**：reshape->assemble reorder 工具遗漏 opcode（如 `OP_SUB`）。触发条件：新增 opcode 未在合并白名单中。检查点：`MergeViewAssembleUtils` 的 opcode 处理。
- **模式 P007**：view/reshape/assemble 合并条件错误。触发条件：多输入/输出、动态 shape、axis-combine pad 轴。检查点：`MergeViewAssembleUtils` 中的 mergeable 谓词。

### 6. 常见被误判为 Pass 问题的症状

| 症状 | 实际常见根因模块 | 快速验证方法 |
|------|------------------|--------------|
| 删除 op 后精度异常 | 上游 `InferTensorFormat` / `AutoCast` 已改变 dtype，本 pass 仅按 shape 判断等价 | 检查被删除 op 的 input/output dtype 是否一致 |
| 删除 op 后 codegen 报错 | `CodegenPreproc` 对特定 pattern 有隐含假设（如需保留某个 reshape） | 在删除前后 dump tile-graph 对比 |
| 冗余 op 未被删除 | 上游 `InsertOpForViewAssemble` 插入了不可删除的 copy 对 | 检查 `notProcessOut_` 与 assembleOutSet |
| 动态 shape 下误删 | `EqualInOut` 中 dynValidShape 比较过严或过松 | 构造 in/out dynValidShape 部分相同的 case |
| 合并后 consumer 找不到输入 | `MergeViewAssembleUtils` 未正确更新 consumer 的 iOperand | 开启 `KEY_ENABLE_PASS_VERIFY` |

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`。

---

### 8. 调试快速入口

- 开启 Pass dump 的命令/环境变量：`CFG_COMPILE_DEBUG_MODE=CFG_DEBUG_ALL`；或配置 `RemoveRedundantOp` 的 `dumpGraph=true`
- 关键中间状态：`operationUpdated`、`iterTime`、`newOps_`、`assembleOutSet_`、`notProcessOut_`
- 推荐先跑的 UT：`framework/tests/.../tile_graph_pass/remove_redundant_op*`；`RemoveRedundantOpChecker` 相关单测
- 相关 checker：`framework/src/passes/pass_check/remove_redundant_op_checker.cpp`、`framework/src/passes/pass_check/merge_view_assemble_checker.cpp`

---

## PreGraphProcess

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | PreGraphProcess |
| 主要源文件 | `framework/src/passes/tile_graph_pass/graph_constraint/pre_graph/pre_graph.{h,cpp}` |
| Pipeline 阶段 | tile_graph；`ReplaceTensor` 之后、`InferDynShape` 之前 |
| 前置依赖 Pass | `ReplaceTensor` |
| 后置消费 Pass | `InferDynShape` |
| 对应 bug 模式 | C002、S004、S009 |

### 2. 设计目标

在 tile graph 进入动态 shape 与子图转换前，统一完成图约束预处理，为后续 checker 建立稳定输入。

### 3. 核心不变量

- producer/consumer 双向边一致，预处理不能遗留悬空 tensor。
- 动态 shape、offset 和属性必须完整保留给 `InferDynShape`。

### 4. 关键设计决策与取舍

该 Pass 是结构性流水线步骤，只允许通过 dump/checker 定界，不能用禁用整个 Pass 的方式规避问题。

### 5. 已知脆弱场景

- graph constraint helper 修改边时未同步反向关系。
- 预处理后动态属性丢失，导致后续 shape 推导失败。

### 6. 常见被误判为 Pass 问题的症状

若原始 shape/属性在进入本 Pass 前已经错误，应优先回查 frontend 和 operation 推导。

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`；额外确认输出满足 `pre_graph_checker`。

### 8. 调试快速入口

- Checker：`framework/src/passes/pass_check/pre_graph_checker.{h,cpp}`。
- UT：`framework/tests/ut/passes/src/test_pre_graph.cpp`。
- 对比本 Pass 前后的边关系、动态 shape 与 op attribute。

---

## InferDynShape

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | InferDynShape |
| 主要源文件 | `framework/src/passes/tile_graph_pass/graph_constraint/infer_dyn_shape.{h,cpp}` |
| Pipeline 阶段 | tile_graph；`PreGraphProcess` 之后、`SubgraphToFunction` 之前 |
| 前置依赖 Pass | `PreGraphProcess` |
| 后置消费 Pass | `SubgraphToFunction` |
| 对应 bug 模式 | C002、S004、S006、S009 |

### 2. 设计目标

在生成 block graph 前推导并固化 tile graph 的动态 shape 约束。

### 3. 核心不变量

- 推导后的动态维、validshape 与 offset 必须可被后续 block graph 转换消费。
- 不得把未知动态维静默替换为无依据的常量。

### 4. 关键设计决策与取舍

这是结构性 Pass；诊断应比较输入/输出 shape 约束并运行 checker，不能建议禁用。

### 5. 已知脆弱场景

- 动态符号未绑定或传播链中断。
- view/reshape/assemble 的 validshape 与满 shape 混用。

### 6. 常见被误判为 Pass 问题的症状

shape 约束来源错误通常来自 frontend 或 operation；先确认进入本 Pass 的输入是否正确。

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`；额外覆盖未知维、空 shape 和 view 链。

### 8. 调试快速入口

- Checker：`framework/src/passes/pass_check/infer_dyn_shape_checker.{h,cpp}`。
- UT：`framework/tests/ut/passes/src/test_infer_shape.cpp`。
- Dump 每个 tensor 的 shape、DynValidShape 与符号绑定。

---

## SubgraphToFunction

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | SubgraphToFunction |
| 主要源文件 | `framework/src/passes/tile_graph_pass/subgraph_to_function.{h,cpp}` |
| Pipeline 阶段 | tile_graph → block_graph leaf function 边界 |
| 前置依赖 Pass | 硬依赖 `GraphPartition`、`ReplaceTensor`、`PreGraphProcess`、`InferDynShape` |
| 后置消费 Pass | block_graph 流水线 |
| 对应 bug 模式 | P020、C002、S004、S009 |

### 2. 设计目标

把完成分区和约束推导的 tile subgraph 转换为 `BLOCK_GRAPH` leaf function。

### 3. 核心不变量

- 每个生成 function 的边界 tensor、参数索引和 op 顺序与原 subgraph 等价。
- 转换后图类型和父子 function 关系必须一致。

### 4. 关键设计决策与取舍

该 Pass 是 tile 到 block 的结构性边界，不能跳过；clone 后属性查找必须支持稳定标识，避免 P020。

### 5. 已知脆弱场景

- clone 后只按 shared pointer 身份查属性。
- 子图边界 tensor 被遗漏或重复注册。

### 6. 常见被误判为 Pass 问题的症状

若分区输入已经错误，应先检查 `GraphPartition`；若 block function 正确但后续失败，再查 block_graph Pass。

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`；额外核对 function 边界、参数索引和 graph type。

### 8. 调试快速入口

- Checker：`framework/src/passes/pass_check/subgraph_to_function_checker.{h,cpp}`。
- UT：`framework/tests/ut/passes/src/test_subgraph_to_function.cpp`、`test_subgraph_to_function_check.cpp`。
- 对比转换前 subgraph 与转换后 leaf function 的边界 tensor 集合。

---
