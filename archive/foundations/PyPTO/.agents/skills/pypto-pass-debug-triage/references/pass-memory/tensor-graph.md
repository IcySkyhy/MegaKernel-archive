# Tensor Graph

## InferTensorFormat

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | InferTensorFormat |
| 所属目录 | `framework/src/passes/tensor_graph_pass/` |
| 主要源文件 | `infer_tensor_format.cpp`、`infer_tensor_format.h` |
| Pipeline 阶段 | tensor_graph |
| 前置依赖 Pass | frontend/operation 构造（已设置 raw tensor format） |
| 后置消费 Pass | `GlobalMemoryReuse`、`SubgraphToFunction`、`GraphPartition`、`GenerateMoveOp`、`AssignMemoryType` 等 |
| 对应 bug 模式 | S001、S004、S016、S034、C002、C005、C007 |

### 2. 设计目标

沿 consumer 链 BFS 推导每个张量的 `TileOpFormat`，在格式不匹配时插入 `OP_TRANSPOSE_MOVEIN/MOVEOUT/VNCHWCONV` 等 TransData 算子，并删除 `OP_FAKE_TRANS`。

### 3. 核心不变量

- 推导完成后每个 tensor 的 `format` 字段必须被赋值。
- `OP_VIEW` / `OP_VIEW_TYPE` / `OP_RESHAPE` 为透传 op，输出 format 等于输入 format。
- `OP_ASSEMBLE` 输出到 function outcast 时强制为 `TILEOP_ND`。
- 仅允许 `IsSupportedTransData` 中列出的转换，否则返回 FAILED。
- 插入 TransData 后必须保留原 tensor 的 `DynValidShape`。
- ND/NZ 兼容格式可互相接受，无需插 TransData。

### 4. 关键设计决策与取舍

| 决策 | 选择 | 原因 | 如果改这里要注意什么 |
|------|------|------|---------------------|
| BFS 起点为 function incast | 从输入张量开始沿 consumer 传播 | 符合数据流方向 | 若 incast format 设置错误，全图推导错误 |
| 透传 op 白名单 | `kPassThroughOps = {OP_VIEW, OP_VIEW_TYPE, OP_RESHAPE}` | 这些 op 不改变物理布局 | 新增透传 op 必须同步，否则 ASSEMBLE 等会被误判 |
| ASSEMBLE 输出 format 决策 | 首个 consumer 初始化后，后续 consumer 复用该 format | 避免重复插 TransData | 多个 ASSEMBLE consumer 要求不同 format 时会静默冲突 |
| FAKE_TRANS 处理 | 先 `EnsureTensorFormat` 到 fakeInFormat，再 `EnsureTensorFormat` 到 fakeOutFormat | 复用现有 TransData 插入逻辑 | 会两次修改 `TileShape::Current()`，产生全局状态副作用 |

### 5. 已知脆弱场景

- **模式 S004**：`kPassThroughOps` 只包含 VIEW/VIEW_TYPE/RESHAPE，ASSEMBLE 单独处理；若新增 view-like opcode 容易遗漏。检查点：所有 `GetOpcode()` 与 view 相关判断处。
- **模式 S016**：`ResolveRequiredInputFormat` 对多个 ASSEMBLE consumer 只取第一个初始化输出 format，冲突场景未被检测。检查点：构造一个张量有两个 ASSEMBLE consumer 且要求不同 format 的 case。
- **模式 S034**：`ApplyTransDataVecTile` 直接修改全局 `TileShape::Current()`，影响 AutoCast / InferMemoryConflict 的 tile/raw size 判断。检查点：TransData 后 dump tile shape。
- **模式 S001**：`ProcessTensorConsumers` 中 consumer 可能为空或已删除，未全部判空。检查点：空 consumer 图。
- **模式 C002/C005/C007**：format 报错常来自 frontend 原始 format、operation 注册的支持列表或 codegen 的 tile 对齐，而非本 Pass。

### 6. 常见被误判为 Pass 问题的症状

| 症状 | 实际常见根因模块 | 快速验证方法 |
|------|------------------|--------------|
| `Unsupported format conversion` 报错 | frontend 给错原始 format，或 operation 注册表中该 op 不支持该 format（C003） | dump 当前 tensor format 与 `OpcodeManager::GetSupportOpFormatList` 对比 |
| TransData 后 shape/validshape 错位 | `InferMemoryConflict` 插 copy 未带 validshape，或 codegen reshape copy 缺失 validshape（N066） | 检查插入的 TransData 输出 tensor 的 `DynValidShape` |
| NC1HWC0/FRACTAL_Z 运行 misalignment | operation 的 tile verifier 只校尾维（N002） | 检查 `SetHeuristicTileShapes` 输出的 tile 与各维对齐要求 |

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`。

---

### 8. 调试快速入口

- 开启 Pass dump 的命令/环境变量：
  - Python：`pypto.set_options(pass_options={"dump_graph": True, "print_graph": True})`
  - 配置文件：`framework/src/interface/configs/tile_fwk_config.json` 中 `global.pass.default_pass_configs.dump_graph = true`
- 关键中间状态：`End_TensorGraph.json`（`RemoveRedundantReshape` 还会生成 `Begin_TensorGraph.json`）。
- 推荐先跑的 UT：`framework/tests/ut/passes/src/test_infer_tensor_format.cpp`。
- 相关 checker：无专属 checker，通用检查在 `framework/src/passes/pass_check/checker.cpp`；图 shape/mem 后续由 `framework/src/passes/pass_check/pre_graph_checker.cpp` 校验。

---

## RemoveRedundantReshape

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | RemoveRedundantReshape |
| 所属目录 | `framework/src/passes/tensor_graph_pass/` |
| 主要源文件 | `remove_redundant_reshape.cpp`、`remove_redundant_reshape.h` |
| Pipeline 阶段 | `PVC2_OOO` 策略，tensor graph 阶段 |
| 前置依赖 Pass | `InferTensorFormat` |
| 后置消费 Pass | `AutoCast`、`InferMemoryConflict`、`RemoveUndrivenView` |
| 对应 bug 模式 | P004、S003、S004、S006、S009、S012、S017 |

---

### 2. 设计目标

先通过 `ViewReshapeAssembleReorderUtils` 重排 view/reshape/assemble 链以暴露 matmul 友好布局，再删除输入输出 shape 相同且所有消费者都是 reshape 的冗余 `OP_RESHAPE`。

---

### 3. 核心不变量

- `CheckIOOperands` 强制每个 reshape 有且仅有一个输入和一个输出。
- 包含动态维度 `-1` 的 reshape 必须跳过。
- 仅当 reshape 的所有消费者都是 `OP_RESHAPE` 且输出不是 outcast 时才删除。
- 删除前必须通过 `consumerOp->ReplaceInput(in, out)` 双向更新 consumer 关系。
- PostCheck 禁止出现连续 reshape。

---

### 4. 关键设计决策与取舍

| 决策 | 选择 | 原因 | 如果改这里要注意什么 |
|------|------|------|---------------------|
| 先重排再删除 | 先调用 `ViewReshapeAssembleReorderUtils::ReorderViewReshapeAssemble` | 能产生更多可删除的 reshape | 重排失败必须返回 FAILED |
| 跳过 -1 shape | `CommonUtils::ContainsNegativeOne` | 动态 shape 下无法判断等价性 | 若要删除需引入动态 shape 等价判断 |
| 消费者全为 reshape 才删 | 保留非 reshape 消费者看到的 shape 语义 | 避免数据流语义错误 | 改为“任意 consumer 是 reshape”会断数据流 |

---

### 5. 已知脆弱场景

- **P004**：OPCode 特判缺失。触发点：`ViewReshapeAssembleReorderUtils::TryRecordReshapeAssemble` 显式跳过 `OP_SUB`，但其他新增 opcode 可能落入未处理分支。检查点：列出所有进入 reorder utility 的 opcode。典型报错：精度异常 / 错误合并。
- **S004**：视图类 OP 处理不完整。触发点：utility 仅处理 `OP_VIEW`/`OP_RESHAPE`/`OP_ASSEMBLE`，忽略 `OP_VIEW_TYPE`。检查点：搜索 `OP_VIEW_TYPE`。典型报错：漏优化或 shape 错误。
- **S017**：consumer 关系未同步。触发点：`RemoveReshape` 调用 `consumerOp->ReplaceInput(in, out)`，若 consumer 多输入可能误改其他边。检查点：图不变量检查。典型报错：producer-consumer 不一致。
- **S003**：类型转换未判空。触发点：`std::dynamic_pointer_cast<ViewOpAttribute>` 在 reorder 路径中部分分支只打 warn 继续。检查点：所有 cast 结果判空。典型报错：bad_cast / 空指针。
- **S009**：动态 shape / validshape 处理不当。触发点：`SetMetadataReshapeAttrs` 用 dynShape 写 `validShape`，若 dynShape 不完整会得到空 validshape。检查点：含符号维度的 reshape。典型报错：EMPTY_VALIDSHAPE。

---

### 6. 常见被误判为 Pass 问题的症状

| 症状 | 实际常见根因模块 | 快速验证方法 |
|------|------------------|--------------|
| Reshape 链未被删除 | frontend / `InferTensorFormat`（shape 推导不一致） | 对比前端 shape 与 tensor graph shape |
| Matmul 布局未优化 | codegen / operator（tile 选择） | 检查 `SetHeuristicTileShapes` 输出 |
| 连续 reshape 仍存在 | `SplitReshape` 或上游 Pass 重新插入 | 开启 dump 对比前后图 |
| 动态 reshape 结果错误 | interpreter（N011、N044） | 对比解释器 reshape 与 pass 输出 validshape |

---

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`。

本词条补充检查项：

- [ ] 修改后是否运行 `RemoveRedundantReshapeChecker::DoPostCheck` 检查连续 reshape？

---

### 8. 调试快速入口

- 开启 Pass dump：

  ```python
  pypto.set_pass_config("PVC2_OOO", "RemoveRedundantReshape",
                        pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
  ```

- 关键中间状态：`redundantResapes` 集合、`ViewReshapeAssembleReorderUtils` 的 records、`CheckIOOperands` 失败点。
- 推荐先跑的 UT：`RemoveRedundantReshapeTest.*`（`framework/tests/ut/passes/src/test_remove_redundant_reshape.cpp`），重点关注 `TestReshapeChain`、`TestViewReshapeReorderWithMatmul`、`TestSubReshapeAssembleSkipReorderWithMatmul`。
- 相关 checker：`framework/src/passes/pass_check/remove_redundant_reshape_checker.cpp`、`framework/src/passes/pass_check/remove_redundant_reshape_checker.h`。

---

---

## ViewReshapeAssembleReorder

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | ViewReshapeAssembleReorder（utility 类，未独立注册） |
| 所属目录 | `framework/src/passes/pass_utils/` |
| 主要源文件 | `view_reshape_assemble_reorder_utils.cpp`、`view_reshape_assemble_reorder_utils.h` |
| Pipeline 阶段 | 由 `RemoveRedundantReshape` 在 `PVC2_OOO` tensor graph 阶段调用 |
| 前置依赖 Pass | 无（调用方为 `RemoveRedundantReshape`） |
| 后置消费 Pass | 无（结果返回给 `RemoveRedundantReshape`） |
| 对应 bug 模式 | P004、S002、S003、S004、S006、S009、S012、S017 |

---

### 2. 设计目标

将 `view → reshape` 与 `reshape → assemble` 链改写为“metadata reshape + 单个 view/assemble”，以暴露连续的 matmul 访存模式，并帮助下游删除冗余 reshape。

---

### 3. 核心不变量

- 仅当 function 中存在 `OP_A_MULACC_B` 或 `OP_A_MUL_B` 时才运行。
- incast/outcast shape 必须是 concrete，或只有第 0 维是 `-1`。
- 链式结构必须满足单一生产者和单一消费者（`GetChainMatch`）。
- 轴向折叠组必须是连续区域（`AreCollapsedGroupsContiguous`）。
- 重映射后的 offset 必须落在 base shape 内（`IsRegionWithinShape`）。
- 新 op 必须继承原链的 span 与 scope。

---

### 4. 关键设计决策与取舍

| 决策 | 选择 | 原因 | 如果改这里要注意什么 |
|------|------|------|---------------------|
| 轴向计划构造 | 贪心乘积匹配（`BuildAxisPlan`） | 实现 reshape 维度的双向映射 | 需同时支持 1→N 与 N→1 折叠 |
| 跳过 sub→reshape→assemble | 显式判断 `OP_SUB` | 保留 read-modify-write 依赖 | 其他有状态 opcode 也要加入跳过名单 |
| Fanout/fanin 变体 | 仅在连续性检查失败时启用 | 处理非连续 view/assemble | 会显著增加图复杂度 |
| Scope 兼容性 | `IsScopeCompatible` | 防止跨 CV fusion scope 重排 | 放宽会破坏同步 |

---

### 5. 已知脆弱场景

- **P004**：opcode 处理缺失。触发点：`TryRecordReshapeAssemble` 跳过 `OP_SUB`，其他 elementwise/状态算子可能未进入白名单。检查点：`ProcessOperations` 中列出所有到达的 opcode。典型报错：错误重排 / 精度异常。
- **S002**：单一生产者/消费者假设。触发点：`GetPrecedingViewOp` 要求 `input->GetProducers().size() == 1`，`GetFollowingAssembleOp` 要求 `output->GetConsumers().size() == 1`。检查点：broadcast/多用途输入图。典型报错：漏优化或误重排。
- **S004**：`OP_VIEW_TYPE` 未处理。触发点：只处理 `OP_VIEW`/`OP_RESHAPE`/`OP_ASSEMBLE`。检查点：函数中搜索 `OP_VIEW_TYPE`。典型报错：shape 错误。
- **S006**：整数溢出。触发点：`ProductShape`/`ProductDynShape` 折叠组乘积可能超过 int64。检查点：大折叠组。典型报错：负维度 / 错误 shape。
- **S012**：边界检查不一致。触发点：`IsRegionWithinShape` 使用 `offset + regionShape > baseShape`。检查点：`offset + region == baseShape` 边界情况。典型报错：合法区域被误判。
- **S017**：图关系同步。触发点：`CreateMetadataReshape`/`CreateView`/`CreateAssemble` 创建新边，旧 op 在 `Append*Records` 中标记删除。检查点：`CleanUp` 中 `EraseOperations` + `SortOperations`。典型报错：访问已删除 op。

---

### 6. 常见被误判为 Pass 问题的症状

| 症状 | 实际常见根因模块 | 快速验证方法 |
|------|------------------|--------------|
| Reshape metadata shape 非法 | operation / interpreter（N044、N028） | 检查 `ExecuteOpReshape` 的 size mismatch 处理 |
| View offset 越界 | codegen / IR type inference（N059、N060） | 验证 `DeduceTensorViewType` 的 rank/offset 检查 |
| Matmul 仍未向量化 | codegen / tile shape（S030、N031） | 检查 `SetHeuristicTileShapes` 与生成 CCE |
| 重排后 scope/sync 错误 | `InsertSync` / `OoOSchedule` | 检查生成 tile graph 的 `syncQueue_` 与 event ID |

---

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`。

本词条补充检查项：

- [ ] 修改 `BuildAxisPlan` 后是否验证 1→N、N→1、等长三种折叠场景？

---

### 8. 调试快速入口

- 开启 dump：通过调用方 `RemoveRedundantReshape` 的配置开启：

  ```python
  pypto.set_pass_config("PVC2_OOO", "RemoveRedundantReshape",
                        pypto.PassConfigKey.KEY_DUMP_GRAPH, True)
  ```

- 关键中间状态：`viewReshapeRecords_`、`reshapeAssembleRecords_`、`axisPlan`、`RemapResult`、`BuildAxisPlan` 成功/失败。
- 推荐先跑的 UT：`RemoveRedundantReshapeTest.TestViewReshapeReorderWithMatmul`、`TestReshapeAssembleReorderWithMatmul`、`TestSubReshapeAssembleSkipReorderWithMatmul`。
- 相关 checker：`framework/src/passes/pass_check/remove_redundant_reshape_checker.cpp`、`framework/src/passes/pass_check/merge_view_assemble_checker.cpp`。

---

---

## AutoCast

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | AutoCast |
| 所属目录 | `framework/src/passes/tensor_graph_pass/` |
| 主要源文件 | `auto_cast.cpp`、`auto_cast.h` |
| Pipeline 阶段 | tensor_graph |
| 前置依赖 Pass | `RemoveRedundantReshape` |
| 后置消费 Pass | `InferMemoryConflict`；并与其一起构成 `ExpandFunction` 的硬前置 |
| 对应 bug 模式 | S017、S034、C003、C005、N012、N032 |

### 2. 设计目标

自动为当前平台不支持的 BF16/FP16 算子插入 FP32 中间 CAST，并合并冗余 CAST 链，保证后续 Pass 看到的图 dtype 合法。

### 3. 核心不变量

- Pass 执行后，任何 op 的输入/输出若仍为 BF16/FP16，则该 op 必须在平台支持列表中。
- 插入的 CAST op 模式为 `CAST_NONE`，且 tile shape 裁剪到目标 tensor 的 rank。
- in/outcast 连接的 tensor 被显式追踪（`inCastConnectedTensors_` / `outCastConnectedTensors_`），避免破坏 IO dtype 语义。
- 仅允许 `legalCastPair` 白名单中的类型转换。

### 4. 关键设计决策与取舍

| 决策 | 选择 | 原因 | 如果改这里要注意什么 |
|------|------|------|---------------------|
| 按 opcode 黑名单 | `UNSUPPORT_BF16_OPS` / `UNSUPPORT_FP16_OPS` 及 ARCH35 变体 | 与硬件能力对齐 | 新增 op 必须更新黑白名单，否则可能漏插或错插 cast |
| ARCH35 特殊处理 | DAV_3510 下额外允许 `DT_INT32 -> DT_FP16` | 平台差异 | 修改 `legalCastPair` 必须同步 `AutoCastChecker::DoPostCheck` |
| 输入输出分开处理 | 先替换输入为 FP32，再替换输出为 FP32 | 保证算子内部全 FP32 | 替换时必须同时更新 producer/consumer 双向关系 |
| 冗余 CAST 链合并 | `GetCastChain` + `ShortenChain` | 减少连续 cast | 涉及 `ReplaceInput` / `ReplaceOutput`，容易漏更新 consumer 列表 |

### 5. 已知脆弱场景

- **模式 S017**：`InsertBF16Cast` / `InsertFP16Cast` 中调用 `op->ReplaceInput(newInput, iop)` 和 `op->ReplaceOutput(newOutput, oop)`；`ShortenChain` 中也大量修改 producer/consumer。检查点：图连接一致性。
- **模式 S034**：插入 CAST 改变了 dtype，下游 `InferMemoryConflict` 的 raw size 判断依赖 dtype；同时依赖 `TileShape` 来自原 op。检查点：cast 后 dump dtype 与 tile shape。
- **模式 C003**：某 op 对 BF16/FP16 不支持时，可能是 operation 层确实无实现，而非本 Pass bug。
- **模式 C005/N012/N032**：IR/frontend 传入错误 dtype、interpreter dtype switch 缺 default、或 `SetOpAttribute` 类型转换错误，都会表现为 AutoCast 后仍报错。

### 6. 常见被误判为 Pass 问题的症状

| 症状 | 实际常见根因模块 | 快速验证方法 |
|------|------------------|--------------|
| `Exist unsupported BF16 compute` | `AutoCastChecker` 正确报错；根因是 operation 不支持该 op 的 BF16（C003） | 查 `OpcodeManager` 支持表与平台 arch |
| 插 cast 后精度仍错 | codegen 未读取 `CAST_NONE` attr（N061）或 interpreter 阈值公式问题（N054） | 对比关闭 AutoCast 的 golden |
| 新增 FP32 tensor 缺 `DynValidShape` | `CreateFp32TensorLike` 已复制，但下游 `InferDynShapeChecker` 仍失败 | 检查动态 shape 是否在 cast 链中正确传递 |

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`。

---

### 8. 调试快速入口

- 开启 Pass dump 的命令/环境变量：
  - Python：`pypto.set_options(pass_options={"dump_graph": True, "print_graph": True})`
  - 配置文件：`framework/src/interface/configs/tile_fwk_config.json` 中 `global.pass.default_pass_configs.dump_graph = true`
- 关键中间状态：观察插入的 `OP_CAST` 的 source/target dtype 与 `CAST_NONE` attr。
- 推荐先跑的 UT：`framework/tests/ut/passes/src/test_auto_cast.cpp`。
- 相关 checker：`framework/src/passes/pass_check/auto_cast_checker.cpp`。

---

## InferMemoryConflict

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | InferMemoryConflict |
| 所属目录 | `framework/src/passes/tensor_graph_pass/` |
| 主要源文件 | `infer_memory_conflict.cpp`、`infer_memory_conflict.h` |
| Pipeline 阶段 | tensor_graph |
| 前置依赖 Pass | `AutoCast`（直接策略前驱）；`pass_dependency.cpp` 将本 Pass 与 `AutoCast` 一起声明为 `ExpandFunction` 的硬前置 |
| 后置消费 Pass | `RemoveUndrivenView`、`ExpandFunction` |
| 对应 bug 模式 | S001、S002、S004、S006、S009、S014、S015、C001、C002、C004 |

### 2. 设计目标

通过正向（从 incast）和反向（从 outcast）传播，判断 producer/consumer 之间是否存在内存冲突；对冲突的 reshape/view 路径插入 `OP_REGISTER_COPY`，并调用 `FunctionUtils::InferOutcastWriteConflict` 为 machine 标记 outcast 冲突。

### 3. 核心不变量

- `memoryInfo` 中每个被传播 tensor 都映射到一个根 memory 符号（初始为 incast/outcast 自身）。
- `CheckConflict` 只有在两个 tensor 的 symbol 不同且 memoryId 不同才判定为冲突。
- 插入的 `OP_REGISTER_COPY` 必须具有与 producer/consumer 兼容的 `TileShape`。
- `viewTypeTable` 必须覆盖当前所有可能参与 VIEW_TYPE 的 dtype（int8/bf16/fp16/fp32/fp8 等）。
- 动态 raw shape 必须保守触发冲突，不能 silently 跳过。
- 优化的 reshape pattern 仅支持 2D/3D/4D/5D 且元素总数相等的情形。

### 4. 关键设计决策与取舍

| 决策 | 选择 | 原因 | 如果改这里要注意什么 |
|------|------|------|---------------------|
| 双向传播 | 正向从 incast，反向从 outcast | 同时捕获输入侧和输出侧的冲突 | 两个方向对同一条边可能给出不同 copy 位置，需保证不重复/遗漏 |
| CheckTransmit 透传集合 | `OP_VIEW/ASSEMBLE/RESHAPE/INDEX_OUTCAST/VIEW_TYPE/ATOMIC_RMW` 视为透明 | 这些 op 不改变内存符号 | 新增 opcode 必须加入集合，否则传播中断 |
| Reshape 优化 pattern | 仅当 producer 是 VIEW/MatMul/ADD/REDUCE_ACC 且 consumer 是 MatMul/ASSEMBLE/VIEW 时才可跳过 copy | 覆盖常见 BMM 场景 | 动态 shape、NZ format、多 consumer 会触发保守插 copy |
| VIEW_TYPE raw size换算 | 通过 `viewTypeTable` 按 dtype 字节数缩放 raw size | 处理同一段内存不同 dtype 解释 | 新增 dtype 必须更新 table，否则 `CheckRawShapeConflict` 错误 |

### 5. 已知脆弱场景

- **模式 S001/S002**：`CheckRawShapeConflict` 直接 `*inTensor->GetConsumers().begin()`；多处假设 producer/consumer 数量为 1。检查点：空/多 producer、空/多 consumer 图。
- **模式 S004**：`HasOnlyViewConsumers` 只检查 `OP_VIEW`，对 `OP_RESHAPE` / `OP_ASSEMBLE` 处理不一致。检查点：view/reshape/assemble 混合链。
- **模式 S006**：`AccumulateRawShapeSize` 中 `rawSize *= shape[i]` 可能 int64 溢出。检查点：超大 shape。
- **模式 S009**：`InsertPrecededCopys`/`InsertPostCopys` 创建新 tensor 时若未正确传递 `DynValidShape`，下游 `InferDynShapeChecker` 会失败。检查点：动态 shape 下 copy 后的 validshape。
- **模式 S014**：`preregcopys` / `postregcopys` 是 `std::set<Operation*>`，迭代顺序依赖指针值。检查点：多次运行比较插入 copy 的顺序。
- **模式 S015**：`CheckConflict` / `IsNoOverlap` 等命名与实际语义必须一致。检查点：阅读函数体并与调用方预期对比。

### 6. 常见被误判为 Pass 问题的症状

| 症状 | 实际常见根因模块 | 快速验证方法 |
|------|------------------|--------------|
| `address conflict` / `memory overlap` | `AssignMemoryType` 给不同 live range 分配了相同 memoryId，或 machine workspace 复用冲突（N006/N009） | 对比本 Pass 推断的冲突点与 `AssignMemoryType` 输出 memoryId |
| BMM NZ 动态轴报错 | operation/interpreter 对动态 shape / view 的处理（N028/N029） | 查看 `FindNegativeDimAfterFirstNzIncast` 报错前的 tensor format |
| `vecTypeTile tile dim n is not even` | `AutoCast` 改变 dtype 后 `UpdateViewTypeTileShape` 未重新对齐 tile（S034） | 检查 VIEW_TYPE 前后 dtype size 比与 tile 尾维 |

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`。

---

### 8. 调试快速入口

- 开启 Pass dump 的命令/环境变量：
  - Python：`pypto.set_options(pass_options={"dump_graph": True, "print_graph": True})`
  - 配置文件：`framework/src/interface/configs/tile_fwk_config.json` 中 `global.pass.default_pass_configs.dump_graph = true`
- 关键中间状态：每个 pass 前后 `.json`/`.tifwkgr`，重点观察 `OP_REGISTER_COPY` 插入位置。
- 推荐先跑的 UT：`framework/tests/ut/passes/src/test_infer_memory_conflict.cpp`。
- 相关 checker：无专属 checker；动态 shape 由 `framework/src/passes/pass_check/infer_dyn_shape_checker.cpp` 校验。

---

## ExpandFunction

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | ExpandFunction |
| 所属目录 | `framework/src/passes/tensor_graph_pass/` |
| 主要源文件 | `expand_function.cpp`、`expand_function.h` |
| Pipeline 阶段 | tensor_graph；`COMPILE_STAGE == CS_TENSOR_GRAPH` 时在此 Pass 后终止 |
| 前置依赖 Pass | 硬依赖 `AutoCast`、`InferMemoryConflict`；直接策略前驱为 `RemoveUndrivenView` |
| 后置消费 Pass | 直接策略后继为 `MergeViewAssemble`；`AssignMemoryType` 是后续 tile_graph 阶段消费者 |
| 对应 bug 模式 | S004、S009、S013、S017、P009、C005、C007 |

### 2. 设计目标

将静态 tensor graph 按每个 op 的 tile shape 展开为 tile graph：重置原 op 的 producer/consumer 关系，逐个调用 tile framework 的 `ExpandOperationInto`，最后把 function 的 `GraphType` 从 `TENSOR_GRAPH` 改为 `TILE_GRAPH`。

### 3. 核心不变量

- 执行后 `function.GetGraphType() == GraphType::TILE_GRAPH`。
- 不存在 operation 环（`OperationLoopCheck` 通过）。
- `ScopeInfo` 一致：`scopeId == -1` 时 `allowParallelMerge` / `allowCrossScopeMerge` 必须为 false；CV 分离平台不能混用 AIC/AIV。
- 非展开 op（`OP_VIEW`、`OP_ASSEMBLE`、`OP_NOP`、`OP_ATOMIC_RMW`）必须保留原属性、scope、OOO scope、token。
- 其他 op 展开后必须重建正确的 producer/consumer 关系。

### 4. 关键设计决策与取舍

| 决策 | 选择 | 原因 | 如果改这里要注意什么 |
|------|------|------|---------------------|
| 先清空再重建 | `ClearIOOperand` 清空所有 tensor 的 producer/consumer | 避免旧关系干扰展开后的新图 | 必须保证 `ProcessForNotExpandOp` / `ExpandOperation` 重建所有关系 |
| 非展开 op 直接复制 | `kNotNeedExpandOps` 透传属性 | 视图类 op 在 tile graph 保持语义 | 新增透传 op 必须同步 `kNotNeedExpandOps` 并复制对应属性 |
| 复制属性白名单 | `CopyAttrFrom(*op, OP_EMUOP_PREFIX)`，再单独复制 `inplaceIdx`、`rmwMode`、`result_token_`、`tokens_` | 保留前端语义 | 新增可选属性容易遗漏（P009） |
|  lightweight topo sort | 展开后调用 `SortOperations(SortOperationsMode::LIGHTWEIGHT)` | 保证基本顺序 | 若展开引入隐式依赖，可能排序不正确 |

### 5. 已知脆弱场景

- **模式 S004**：`kNotNeedExpandOps` 只处理 VIEW/ASSEMBLE/NOP/ATOMIC_RMW，RESHAPE 被展开；若展开逻辑对 view-like op 处理不一致会出问题。检查点：view/reshape/assemble 在展开后的属性。
- **模式 S009**：展开创建的新 tensor/op 可能丢失 `DynValidShape` 或 dyn offset。检查点：动态 shape 展开后 `InferDynShapeChecker`。
- **模式 S013**：clone/split 后 `result_token_`、`tokens_`、dynParam 索引未按新 function 的 operand layout 重映射。检查点：多 call op / 动态参数 case。
- **模式 P009**：`ProcessForNotExpandOp` 只复制部分属性，新增属性会丢失。检查点：对比原 op 与 clone op 的属性列表。
- **模式 S017**：重建图时 `PassOperationUtils::AddOperation` 与 `ClearIOOperand` 必须双向更新 producer/consumer。检查点：展开后 `OperationLoopCheck` 与 `CheckConsumerRelation`。

### 6. 常见被误判为 Pass 问题的症状

| 症状 | 实际常见根因模块 | 快速验证方法 |
|------|------------------|--------------|
| `Operation Loop detected after expand function` | 前端/IR 图已存在环，或 `RemoveUndrivenView` 未清理死边 | 在 ExpandFunction 前运行 `function.OperationLoopCheck()` |
| Scope/CV mix 报错 | frontend 错误使用 `sg_set_scope`（C005） | 检查 `VerifyScopeInfo` 输出 |
| 展开后属性缺失 | `CopyAttrFrom` 白名单遗漏（P009） | 对比展开前后 op 的 `DumpJson()` |

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`。

---

### 8. 调试快速入口

- 开启 Pass dump 的命令/环境变量：
  - Python：`pypto.set_options(pass_options={"dump_graph": True, "print_graph": True})`
  - 配置文件：`framework/src/interface/configs/tile_fwk_config.json` 中 `global.pass.default_pass_configs.dump_graph = true`
- 关键中间状态：`End_TensorGraph.json`（ExpandFunction 导出）、各 op 展开前后的 `.tifwkgr`。
- 推荐先跑的 UT：`framework/tests/ut/passes/src/test_expand_function.cpp` + `framework/tests/ut/passes/src/ut_json/expand_function.json`。
- 相关 checker：`framework/src/passes/pass_check/expand_function_checker.cpp`。

---

## SetHeuristicTileShapes

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | SetHeuristicTileShapes |
| 所属目录 | `framework/src/passes/tensor_graph_pass/` |
| 主要源文件 | `set_heuristic_tile_shapes.cpp`、`set_heuristic_tile_shapes.h` |
| Pipeline 阶段 | tensor_graph（未在 `RegPass()` 默认流水线注册，通常由自定义策略显式调用或做 standalone tile-shape 研究） |
| 前置依赖 Pass | `InferTensorFormat`（已确定 format）、frontend/operation 构造的 shape/dtype |
| 后置消费 Pass | `InferMemoryConflict`、`NBufferMerge`、`L1CopyInReuseMerge`、`CodegenPreproc` 等读取 `TileShape` 的 Pass |
| 对应 bug 模式 | S001、S006、S008、S030、S031、S032、S033、C007 |

### 2. 设计目标

为 function 中每个 operation 的输入/输出张量生成启发式 Cube/Vector TileShape：Cube 侧搜索满足 L0A/L0B/L0C 容量约束的 `(m,k,n)` 组合并打分，Vector 侧从 Cube/Reduce/无消费者节点出发做 BFS 传播，最终输出可直接被 codegen 与内存规划使用的 tile 配置。

### 3. 核心不变量

- 每个 op 执行后 `op.GetTileShape().GetVecTile()[0] != -1`，即所有 tile 必须被设置。
- Cube tile 必须满足 `m*k*inputTypeSize <= L0A_MAX_SIZE/DOUBLE_BUFFER`、`k*n*inputTypeSize <= L0B_MAX_SIZE/DOUBLE_BUFFER`、`m*n*outputTypeSize <= L0C_MAX_SIZE/DOUBLE_BUFFER`。
- Vector tile 最后一维必须 32 字节对齐且不超过 UB 预算（`MIN_TILE_SIZE`~`MAX_TILE_SIZE`）。
- `TileShape` 的 rank 必须与对应张量的 shape rank 一致。
- 动态维度 `-1` 不得进入 `std::gcd` / `std::log2` 路径。

### 4. 关键设计决策与取舍

| 决策 | 选择 | 原因 | 如果改这里要注意什么 |
|------|------|------|---------------------|
| Cube/Vector 分阶段开关 | 通过头文件宏 `#define CUBE_TILES` / `#define VECTOR_TILES` 控制 | 方便单独关闭某一路进行调试 | 修改宏会影响 `SetHeuristicTileShapesFunc` 的整体输出，下游 `InferMemoryConflict` 可能拿到空 tile |
| Cube tile 搜索策略 | 以 2 的幂枚举到 shape 边界，再按 L0 利用率、任务数、平衡性、L1 copy 周期打分 | 兼顾算力利用与并行度 | `ShapeAndTypeSetting` 的形参顺序是 `(op, shapeM, shapeK, shapeN)`，调用时容易把 K/N 传反 |
| Vector tile 传播顺序 | 先 Cube 反向/正向，再无消费者节点反向，再 Reduce 正反向 | 让计算密集型节点决定周边 tile | 若图中存在环，`FordBellman` / BFS 可能不收敛，需前置环检测 |
| 平台常量硬编码 | `L0A_MAX_SIZE`、`L0B_MAX_SIZE`、`L0C_MAX_SIZE`、`CUBE_CORES`、`BLOCK_SIZE`、`BYTES_PER_REPEAT` 等直接取自 `Platform::Instance()` 或头文件常量 | 减少运行时计算 | 新增平台时需要同步这些常量，否则 tile 可能超限 |

### 5. 已知脆弱场景

- **模式 S030**：`UniqueTilesFilling` 中调用 `ShapeAndTypeSetting(&op, shapeM, shapeN, shapeK)`，但函数签名为 `(op, shapeM, shapeK, shapeN)`，shapeK/N 被交换，导致相同 matmul 的 tile 分组错误。检查点：核对所有 `ShapeAndTypeSetting` 调用点与签名。
- **模式 S031**：`TileSizeCalculation`、`MaxInputShapeCalculation` 中检测到 `maxTypeSize==0` 或 `inputTypeSize==0` 后只打印 `APASS_LOG_ERROR_F`，继续执行会导致除零。检查点：构造 dtype 非法或 `BytesOf` 返回 0 的 case。
- **模式 S032**：`ReshapeTileSetting` 中 `std::gcd(curTile, maxInputShape[...])` 与 `std::log2(tileSize)` 未处理动态维 `-1` 或 `<=0`。检查点：构造含 `-1` 的动态 shape。
- **模式 S033**：`FordBellman` 与 BFS 传播沿 producer/consumer 递归/迭代时无 visited 保护，环图会栈溢出或死循环。检查点：构造含环图。
- **模式 S001/S006/S008**：`FillNoConsumersOperations`、`MaxInputShapeCalculation` 中直接 `[0]` 解引用空输入；shape 乘积累加存在 int64 溢出风险；大量平台常量硬编码。检查点：空输入、超大 shape、跨平台验证。

### 6. 常见被误判为 Pass 问题的症状

| 症状 | 实际常见根因模块 | 快速验证方法 |
|------|------------------|--------------|
| 编译期 `INVALID_TILE` / `AICore misalignment` | codegen / operation 的 `CheckCubeTile` / `CheckL1L0Tile`（N002/N031） | 对比 `python_tiles.json` 中的 cube tile 与 codegen 对齐要求 |
| 运行期 workspace 过大或 `rtMalloc failed` | machine 的 `workspace_budget_calculator`（N006）按 tile 数高估并行度 | 检查 workspace 日志与 `max_workspace_kb` |
| reshape/transpose 后结果错位 | `InferMemoryConflict` 漏插/多插 copy，或 frontend 原始 shape 错误（C002） | dump `End_TensorGraph.json` 观察 copy 位置与 validshape |

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`。

---

### 8. 调试快速入口

- 开启 Pass dump 的命令/环境变量：
  - Python：`pypto.set_options(pass_options={"dump_graph": True, "print_graph": True})`
  - 配置文件：`framework/src/interface/configs/tile_fwk_config.json` 中 `global.pass.default_pass_configs.dump_graph = true`
  - C++ UT：`config::SetPassConfig(KEY_DUMP_GRAPH, true); config::SetPassConfig(KEY_PRINT_GRAPH, true);`
- 关键中间状态：`python_tiles.json`、`semantic_labels_tiles.json`（位于 `config::LogTopFolder()`），以及 `output/pass/<func>/` 下的 `.json` / `.tifwkgr` 文件。
- 推荐先跑的 UT：`framework/tests/ut/passes/src/test_set_heuristic_tile_shapes.cpp`（`TestCube`、`TestVector`）。
- 相关 checker：无专属 checker，通用检查在 `framework/src/passes/pass_check/checker.cpp`；动态 shape 后续由 `framework/src/passes/pass_check/infer_dyn_shape_checker.cpp` 校验。

---

## LoopUnroll

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | LoopUnroll |
| 所属目录 | `framework/src/passes/tensor_graph_pass/` |
| 主要源文件 | `loop_unroll.cpp`、`loop_unroll.h` |
| Pipeline 阶段 | tensor_graph / dynamic-loop-specific；默认策略 `FunctionUnroll` 单独调用 |
| 前置依赖 Pass | `CopyOutResolve`、`SrcDstBufferMerge` |
| 后置消费 Pass | `DynAttrToStatic`、`InferDiscontinuousInput`、`MixSubgraphSplit` 等 |
| 对应 bug 模式 | S004、S009、S013、S015、S033、C004、C005、N041 |

### 2. 设计目标

把配置在 `CONVERT_TO_STATIC` 中的动态循环（`FunctionType::DYNAMIC_LOOP` / `DYNAMIC_LOOP_PATH`）展开为静态 tensor graph：遍历 call op，克隆局部 tensor 到全局 top function，计算动态符号的静态值，并处理跨迭代 WAW/WAR 依赖。

### 3. 核心不变量

- 每个被展开函数的 incast/outcast 必须且只能有一个 slot（`GetInCastSlot().size() == 1`）。
- 局部 tensor 必须通过 `tensorLocal2Global` 映射到全局 tensor，不能残留局部 magic。
- 克隆 op 的 `DynValidShape`、`view/assemble offset` 必须被 `UpdateCloneOpAttributes` 正确替换为静态值。
- 跨迭代的 WAW 必须通过 `IsNoOverlapWAW` / `IsTensorOverlap` 判断；重叠时必须新建全局 tensor。
- `lastWriteMap_` 在每个 loop iteration 开始时由 `UpdateGlobalTensorWAW` 标记跨迭代状态。

### 4. 关键设计决策与取舍

| 决策 | 选择 | 原因 | 如果改这里要注意什么 |
|------|------|------|---------------------|
| 按 callop 树递归展开 | `TraverseCallOp` -> `TopFunctionUnroll` -> `ExpandDynamicFunction` | 支持嵌套动态函数 | 每层都要正确更新 callee hash 与 caller-callee link |
| 全局 tensor 复用 | `lastWriteMap_` 按 slot 记录上次写入，结合 WAW/WAR 判断是否复用 | 减少内存分配 | `FindSlotDepend` 递归遍历 consumer 链，环图会无限递归 |
| 局部->全局 clone | `tensor->Clone(*topFunction_, true)` | 保留 raw tensor 与属性 | clone 后必须同步更新 view/assemble 的 offset/validshape |
| 新建 wrapper loop function | `CreateLoopFunc` / `CreateLoopUnrollFunc` | 维持原 function 入口不变 | 新的 `FunctionType` / `GraphType` 必须设置正确，否则后续 Pass 拒绝 |

### 5. 已知脆弱场景

- **模式 S004**：`EvaluateDynamicOpParams` / `UpdateCloneOpAttributes` 对 `OP_VIEW` / `OP_ASSEMBLE` 做 dynamic_pointer_cast，若新增 view-like opcode 会遗漏。检查点：动态展开后的 view/reshape/assemble 链。
- **模式 S009**：动态 shape / dyn offset 在 `UpdateCloneOpAttributes` / `UpdateOutTensorDynAttributes` 中若未找到对应 map 会保留空值。检查点：构造动态 loop 用例。
- **模式 S013**：展开后 `callOpAttr->SetCalleeHash`、scope、token 等索引必须按新 function 的 operand layout 重映射。检查点：多参数 call op。
- **模式 S015**：`IsTensorOverlap` 语义与命名相反（返回 `true` 表示无重叠），容易误导。检查点：阅读函数体与调用方。
- **模式 S033**：`FindSlotDepend` 沿 consumer 链递归无 visited，环图会栈溢出。检查点：构造循环依赖图。
- **模式 C004/C005/N041**：运行期 dynamic loop launch 失败、symbol 求值错误、或 `GetDynloopAttribute()` 空解引用常被误判为展开问题。

### 6. 常见被误判为 Pass 问题的症状

| 症状 | 实际常见根因模块 | 快速验证方法 |
|------|------------------|--------------|
| `Incast/Outcast has multi slot` | frontend/IR 生成多 slot IO | 检查 `function.GetInCastSlot` / `GetOutCastSlot` |
| 动态 loop 边界错误或死循环 | operator/frontend 构造 `LoopRange` 时未校验零/整除（N083） | 打印 `EvaluateSymbolicScalar(loop->Begin/End/Step)` |
| WAW overlap 误判 | operator/frontend 计算的 assemble offset 错误（N084） | 对比 `opDynOffsetMap` 与 `AssembleOpAttribute::GetToOffset()` |

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`。

---

### 8. 调试快速入口

- 开启 Pass dump 的命令/环境变量：
  - Python：`pypto.set_options(pass_options={"dump_graph": True, "print_graph": True})`
  - 配置文件：`framework/src/interface/configs/tile_fwk_config.json` 中 `global.pass.default_pass_configs.dump_graph = true`
- 关键中间状态：展开后的 top function `.json`/`.tifwkgr`；观察 call op 是否被替换、动态符号是否已求值。
- 推荐先跑的 UT：`framework/tests/ut/passes/src/test_loop_unroll.cpp`、`framework/tests/ut/interface/src/interpreter/test_interp_loop_unroll_if.cpp`。
- 相关 checker：无专属 checker；动态 attr 与 shape 由 `framework/src/passes/pass_check/expand_function_checker.cpp` 与 `framework/src/passes/pass_check/infer_dyn_shape_checker.cpp` 共同覆盖。

---
