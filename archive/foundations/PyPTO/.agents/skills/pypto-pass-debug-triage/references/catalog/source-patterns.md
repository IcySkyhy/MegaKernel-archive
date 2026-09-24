# Pass Source Pattern Index

> 本文件是 `../patterns/source-patterns.json` 的人类可读索引。修改结构化条目时必须同步更新本表。

无历史提交的条目只表示源码审计假设，不能当作已确认缺陷。

| ID | 类别 | Pass / 组件 | 历史提交 | 证据状态 |
|----|------|-------------|----------|----------|
| S001 | 源码模式：空容器解引用 | AssignMemoryType<br>SpillBuffer<br>OoOScheduler<br>SplitLargeFanoutTensor<br>ReduceCopy<br>TileGraphPass | `d08d676a` | 历史修复 |
| S002 | 源码模式：多生产者/消费者只取第一个 | AssignMemoryType<br>SplitLargeFanoutTensor<br>ReduceCopy<br>MixSubgraphSplit<br>TileGraphPass | `090e0042`<br>`fedab26e`<br>`513f875f` | 历史修复 |
| S003 | 源码模式：类型转换未判空 | AssignMemoryType<br>SplitLargeFanoutTensor<br>InsertOpForViewAssemble | `b65479d1` | 历史修复 |
| S004 | 源码模式：视图类 OP 处理不完整 | SupernodeGraphBuilder<br>ReduceCopy<br>SpillBuffer<br>MixSubgraphSplit<br>ExpandFunction<br>InferMemoryConflict | `336914aa`<br>`09ac5e3d`<br>`6d0f7055` | 历史修复 |
| S005 | 源码模式：内存类型推断边界过窄 | AssignMemoryType<br>ReduceCopy<br>SplitLargeFanoutTensor<br>TileGraphPass | `a0d4151f`<br>`957571de`<br>`76d3acda` | 历史修复 |
| S006 | 源码模式：整数溢出 | AssignMemoryType<br>SplitLargeFanoutTensor<br>InferMemoryConflict<br>ReduceCopy | `6a262d5b` | 历史修复 |
| S007 | 源码模式：哈希/顺序敏感 | ReduceCopy<br>SupernodeGraphBuilder<br>TileGraphPass<br>MixSubgraphSplit | `8a99ad83`<br>`82e880fc` | 历史修复 |
| S008 | 源码模式：硬编码常量/Magic Number | AssignMemoryType<br>ReduceCopy<br>SupernodeGraphBuilder<br>InferMemoryConflict<br>TileGraphPass | `c6042765` | 历史修复 |
| S009 | 源码模式：动态 shape / validshape 处理不当 | SplitLargeFanoutTensor<br>SpillBuffer<br>InferMemoryConflict<br>TileGraphPass | `6d0f7055`<br>`3a7f8568` | 历史修复 |
| S010 | 源码模式：统计字段下溢/累加错误 | SpillBuffer<br>OoOScheduler | `3e379fa0` | 历史修复 |
| S011 | 源码模式：返回值未检查 | SpillBuffer<br>MixSubgraphSplit | `b5d3611a` | 历史修复 |
| S012 | 源码模式：边界检查范围不一致 | SupernodeGraphBuilder<br>SplitLargeFanoutTensor | `b8cca837` | 历史修复 |
| S013 | 源码模式：拆分/克隆后 offset/dynParam 未重映射 | MixSubgraphSplit | `b5d3611a` | 历史修复 |
| S014 | 源码模式：指针有序容器导致非确定性 | InferMemoryConflict<br>SupernodeGraphBuilder | `8a99ad83` | 历史修复 |
| S015 | 源码模式：命名与语义相反/逻辑倒置 | LoopaxesProc<br>InferMemoryConflict<br>SplitLargeFanoutTensor | `b8cca837` | 历史修复 |
| S016 | 源码模式：多消费者格式冲突只取第一个 | InferTensorFormat | `5edc4c44` | 历史修复 |
| S017 | 源码模式：删除/替换 op 时 consumer 关系未同步 | RemoveRedundantOp<br>RemoveRedundantReshape<br>SplitLargeFanoutTensor | `f5d8ca0d` | 历史修复 |
| S018 | 源码模式：OSP 空处理器类型向量解引用 | OspPartitioner<br>IsomorphicSubgraphScheduler | — | 源码审计假设 |
| S019 | 源码模式：OSP 通信代价 staleness 下溢 | OspPartitioner | — | 源码审计假设 |
| S020 | 源码模式：OSP 图适配器持有外部 vector 裸指针 | OspPartitioner | — | 源码审计假设 |
| S021 | 源码模式：OSP 分区权重窄化转换溢出 | OspPartitioner | — | 源码审计假设 |
| S022 | 源码模式：Merkle 哈希碰撞导致误判同构 | OspPartitioner<br>IsomorphicSubgraphScheduler | — | 源码审计假设 |
| S023 | 源码模式：BSP 处理器数量累加溢出 | OspPartitioner | — | 源码审计假设 |
| S024 | 源码模式：兼容矩阵维度不匹配访问 | OspPartitioner | — | 源码审计假设 |
| S025 | 源码模式：KL 调度移动操作缺少边界校验 | OspPartitioner | — | 源码审计假设 |
| S026 | 源码模式：Pass dump 子函数指针未判空 | PassInterface | — | 源码审计假设 |
| S027 | 源码模式：PassManager startIdx 跨策略污染 | PassManager | — | 源码审计假设 |
| S028 | 源码模式：BuildHashValues 反向循环在无算子时下溢 | OspPartitioner | — | 源码审计假设 |
| S029 | 源码模式：UpdatePartitionResult 超点映射不一致 | OspPartitioner | — | 源码审计假设 |
| S030 | 源码模式：Cube Tile 参数顺序错位 | SetHeuristicTileShapes | — | 源码审计假设 |
| S031 | 源码模式：仅打日志不阻断的除零路径 | SetHeuristicTileShapes | — | 源码审计假设 |
| S032 | 源码模式：动态维 / -1 进入 gcd/log2 计算 | SetHeuristicTileShapes | — | 源码审计假设 |
| S033 | 源码模式：递归/有环图无访问标记 | LoopUnroll<br>SetHeuristicTileShapes | — | 源码审计假设 |
| S034 | 源码模式：跨 Pass 状态依赖与全局可变状态 | InferTensorFormat<br>AutoCast<br>InferMemoryConflict | — | 源码审计假设 |
| S035 | 源码模式：数组/Vector 越界索引 | DynAttrToStatic<br>InferParamIndex<br>TuneSyncForVF<br>TuneTileOpSeqForVF | — | 源码审计假设 |
| S036 | 源码模式：std::map::operator[] 默认构造导致空指针解引用 | TuneSyncForVF | — | 源码审计假设 |
| S037 | 源码模式：IfStmt/ForStmt body 被默认为 SeqStmts | MergeStmts<br>InferToken<br>ConvertToSSA | — | 源码审计假设 |
| S038 | 源码模式：SSA 转换器对 IterArg 名称做字符串截取 | ConvertToSSA | — | 源码审计假设 |
| S039 | 源码模式：SubstituteVarsInType 只改写 validShape | ConvertToSSA | — | 源码审计假设 |
| S040 | 源码模式：SSA 循环携带变量构造中使用未校验的 unordered_map::at | ConvertToSSA | — | 源码审计假设 |
| S041 | 源码模式：TensorType 的结构相等与哈希忽略 tensor_view 和 memref | StructuralEqual<br>StructuralHash<br>CSE | — | 源码审计假设 |
| S042 | 源码模式：结构哈希的变量身份依赖遍历顺序 | StructuralHash | — | 源码审计假设 |
| S043 | 源码模式：OpConversionRegistry 静默覆盖已有转换 | OpConversionRegistry | — | 源码审计假设 |
| S044 | 源码模式：merge_stmts_pass 的变量替换只处理直接 Var | MergeStmts | — | 源码审计假设 |
| S045 | 源码模式：disable_pass 绕过必要 Pass 依赖 | PassInterface<br>PassManager<br>PassDependency | — | 源码审计假设 |
