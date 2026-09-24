# Pass Historical Pattern Index

> 本文件是 `../patterns/pass-patterns.json` 的人类可读索引。修改结构化条目时必须同步更新本表。

详细反模式、修复方向和验证动作以 `../patterns/pass-patterns.json` 为准。

| ID | 类别 | Pass / 组件 | 历史提交 | Related Issues | 证据状态 |
|----|------|-------------|----------|----------------|----------|
| P001 | 边界/范围检查错误 | LoopaxesProc | `b8cca837` | — | 历史修复 |
| P002 | 函数语义与命名/实现不符 | OoOScheduler<br>SpillBuffer | `3e379fa0` | — | 历史修复 |
| P003 | 累加/赋值错误 | SpillBuffer | `3e379fa0` | — | 历史修复 |
| P004 | OPCode 特判缺失 | RemoveRedundantReshape<br>ViewReshapeAssembleReorder | `336914aa` | #2391 | 历史修复 |
| P005 | validshape 推导错误 | SplitLargeFanoutTensor | `6d0f7055` | — | 历史修复 |
| P006 | dtype/数据搬移路径错误 | OoOScheduler<br>SpillBuffer | `9b89dbd8` | #2428 | 历史修复 |
| P007 | 视图/Reshape 合并条件错误 | SupernodeGraphBuilder<br>GraphPartition | `09ac5e3d`<br>`705ed0f6`<br>`fab94d96`<br>`8248a429`<br>`9ddb6566` | — | 历史修复 |
| P008 | 内存类型/地址推断错误 | AssignMemoryType | `a0d4151f` | — | 历史修复 |
| P009 | 属性拷贝/克隆不完整 | CopyOpAttribute | `b65479d1` | — | 历史修复 |
| P010 | 计算图成环/自环错误 | COE | `2f18692c`<br>`799d057f` | — | 历史修复 |
| P011 | 多生产者/多消费者场景处理错误 | OoOScheduler<br>SpillBuffer | `fedab26e`<br>`090e0042` | — | 历史修复 |
| P012 | 原子操作/同步条件错误 | MVA | `75782528`<br>`aac4c900` | — | 历史修复 |
| P013 | 哈希/排序顺序敏感错误 | ReduceCopy | `8a99ad83` | — | 历史修复 |
| P014 | 调度/事件标识错误 | OoOScheduler | `f40725ad`<br>`7ec4523e` | — | 历史修复 |
| P015 | 直连内存路径回退错误 | AssignMemoryType | `957571de`<br>`76d3acda`<br>`72c7201a` | — | 历史修复 |
| P016 | 新建中间 Tensor 丢失 DynValidShape | InsertOpForViewAssemble<br>MergeViewAssemble<br>ReplaceTensor | `af20e717` | — | 历史修复 |
| P017 | Reshape->Assemble 重排破坏 validShape | ViewReshapeAssembleReorder<br>RemoveRedundantReshape | `4a067951`<br>`336914aa` | #2518 | 历史修复 |
| P018 | Axis-combine 下 shape-transform OP pad 轴错误 | PadLocalBuffer<br>AxisCombineMarker | `fab94d96` | #2458 | 历史修复 |
| P019 | 公共算子消除误删语义敏感 OP | CommonOperationEliminate<br>ReduceCopyMerge | `72fe6828` | #2476 | 历史修复 |
| P020 | MixSubgraphSplit 属性查找只按指针身份 | MixSubgraphSplit<br>SubgraphToFunction | `b5d3611a` | #2502 | 历史修复 |
| P021 | ReduceCopy 合并后内部 Tensor 仍被外部子图使用 | ReduceCopyMerge<br>GraphPartition<br>SupernodeGraphBuilder | `43cb5627`<br>`9668b4f4` | #2279<br>#2365 | 历史修复 |
| P022 | 动态轴默认 tile 推导溢出或卡死 | InferMemoryConflict<br>SetHeuristicTileShapes | `90d72725` | — | 历史修复 |
| P023 | InsertSync 事件 ID 放松逻辑误判失败或迭代失效 | InsertSync<br>PipeSync<br>OoOScheduler | `34156240` | — | 历史修复 |
| P024 | SplitRaw 漏处理特殊 copyout 语义 OP | SplitRawTensor<br>Distributed<br>SHMEM | `4437fe36` | — | 历史修复 |
| P025 | CopyOut 级联内存复用后 raw_shape 属性未刷新 | ReplaceTensor<br>CopyOutResolve<br>Hub | `8b3cacc1` | #2072 | 历史修复 |
