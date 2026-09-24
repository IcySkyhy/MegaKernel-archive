# Pass 设计记忆文档 Atlas

> 本文件根据源码和 bug 模式库整理维护，用于快速定位 Pass 设计意图、不变量与常见误判。
> 新增 Pass 或重大重构时，请按 `../../templates/pass-memory-entry.md` 补充或更新。

---

## 查找提示

先按真实 graph stage 选择文件，再搜索标题 `## {PassName}`。内部 helper 与基础设施不伪装成独立注册 Pass。

| 文件 | 范围 |
|------|------|
| `tensor-graph.md` | tensor graph Pass、`ExpandFunction` 边界和相关 helper |
| `tile-graph.md` | tile graph Pass、`SubgraphToFunction` 边界和相关 helper |
| `block-graph.md` | block graph、OoO 调度及 codegen 前处理 Pass |
| `infrastructure.md` | 跨阶段 `PassManager` 与 `Pass` 基类 |

| 注册 Pass / 用户说法 | 优先读取章节 |
|----------------------|--------------|
| `OoOSchedule` / OoO schedule | `OoOScheduler`、`SpillBuffer` |
| `GraphPartition` | `tile-graph.md`：`SupernodeGraphBuilder`（内部 helper）、`OspPartitioner` |
| `AssignMemoryType` | `AssignMemoryType`、`TileAssignMemoryType` |
| `RemoveRedundantReshape` | `RemoveRedundantReshape`、`ViewReshapeAssembleReorder` |
| `RemoveRedundantOp` | `TileRemoveRedundantOp` |
| `InferMemoryConflict` | `InferMemoryConflict`、`SetHeuristicTileShapes` |
| `CommonOperationEliminate` / COE | `tile-graph.md`：`TileRemoveRedundantOp`，并结合 P010 / P019 / S045 |
| `PreGraphProcess` / `InferDynShape` / `SubgraphToFunction` | `tile-graph.md` 中各自独立词条 |
| `AddAlloc` / `RemoveAlloc` / `CopyOutResolve` / `InsertSync` / `CodegenPreproc` | `block-graph.md` 中各自独立词条 |
| `PassManager` / Pass 配置 / `disable_pass` | `infrastructure.md`：`PassManager`、`Pass base class (pass_interface)` |

---
