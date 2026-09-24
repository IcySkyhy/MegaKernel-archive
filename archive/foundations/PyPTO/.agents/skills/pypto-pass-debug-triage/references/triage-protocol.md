# Pass 跨模块定界协议

## 使用时机

当用户只说“Pass 报错”“Pass 精度不对”“Pass 后图错了”时，先读本文件，再进入 Pass 内模式匹配。目标是防止把 operation、machine、frontend、codegen 等模块的问题误留在 Pass。

## 症状到模块速查

| 症状关键词 | 最可能模块 | 优先排查的非 Pass 文件 | 若排除后仍像 Pass |
|-----------|-----------|----------------------|------------------|
| spill / L0C / L1 / copyOut / copyIn / dtype / MTE / sync | pass > machine | `framework/src/machine/runtime/*`, `framework/src/machine/launch/*` | SpillBuffer / AssignMemoryType / OoOScheduler |
| reshape / view / assemble / validshape / shape infer | pass > operation > frontend | `framework/src/interface/operation/*`, `python/pypto/frontend/*` | SupernodeGraphBuilder / SplitLargeFanoutTensor |
| matmul / conv / transdata / relu / gather / where / index_add | operation > pass | `framework/src/interface/operation/*`, `models/*/ops/*` | Pass 是否漏处理该 opcode |
| runtime / launch / rtMalloc / workspace / memory pool / timeout / aicore | machine > pass | `framework/src/machine/memory/*`, `framework/src/machine/runtime/*` | Pass 是否生成不合理图 |
| python / jit / frontend / symbolic / scalar | frontend / front | `python/pypto/frontend/*` | IR 表示是否与预期一致 |
| IR / token / dce / transform / ssa | ir > pass | `framework/src/interface/ir/*` | Pass 是否正确消费 IR 数据 |
| distributed / shmem / allreduce / collective | distributed | `framework/src/distributed/*`, `framework/src/interface/operation/distributed/*` | 分布式 tensor 处理 |
| codegen / tile / unroll / cce / kernel | codegen > pass | `framework/src/codegen/*` | tile shape / memory type 输出 |
| precision / accuracy / fp16 | operation > pass | `framework/src/interface/operation/*` | Pass 是否错误合并、搬移或转换 dtype |

## 决策树

```text
报错位置在 framework/src/passes/ 某个 Pass 内
  |
  +- 症状涉及 spill/L0C/L1/dtype/地址/内存类型
  |  +- 先查 Pass 的 AssignMemoryType / SpillBuffer / OoOScheduler
  |  +- 若涉及运行时同步/事件/launch, 同时查 machine
  |
  +- 症状涉及 reshape/view/assemble/merge/validshape
  |  +- 先查 Pass 的 SupernodeGraphBuilder / SplitLargeFanoutTensor
  |  +- 若 shape 来自前端输入, 查 frontend/front
  |
  +- 症状涉及具体算子
  |  +- 先查 operation/operator 的算子实现与约束
  |  +- 再查 Pass 是否漏处理该 opcode
  |
  +- 症状涉及 runtime/aicore/同步/timeout/memory pool
  |  +- 先查 machine
  |
  +- 症状涉及精度
     +- 先查 operation 算子实现精度
     +- 再查 Pass 是否错误合并、搬移或转换 dtype
```

## 非 Pass 结论必须包含

1. 明确结论：当前证据更支持哪个模块。
2. 定界依据：命中了哪些关键词、路径或历史模式。
3. 推荐先排查的非 Pass 文件：至少 2 个具体路径或目录。
4. 继续分析 Pass 的条件：说明排除哪些非 Pass 证据后再回到 Pass。

## 常见误判

| 现象 | 常见真实根因 | 快速验证 |
|------|-------------|---------|
| `rtMalloc failed` / workspace OOM | machine 内存分配或 Pass 生成异常大 tensor | 对比 pass 前后 tensor size / workspace log |
| `aicore error` / 运行期非法地址 | machine 同步、operator 算子实现、codegen kernel | 定位 CCE 文件，看是 codegen 还是算子指令 |
| 精度异常 | operation 算子精度、frontend 常量转换、量化参数 | 单算子 NPU vs CPU 对比 |
| `std::out_of_range` / `bad_any_cast` | operation 属性构造、IR 类型推导、frontend 传参 | 检查 op attribute / IR 类型 |
| `cycle detected` / topo sort 失败 | Pass 自身或 IR 已含环 | pass 前后环检测 |
| 动态 shape 验证失败 | interpreter 求值、frontend 符号绑定 | dump validshape / symbolic bound |

## Pass 开关诊断规则

- 不要把“关闭某个 Pass 看是否复现”当作默认建议。必须先确认该 Pass 是可禁用优化 Pass。
- `OoOSchedule` / `OoOScheduler` 是必要调度链路，不建议禁用或绕过。怀疑它时，改用 `dump_graph`、`health_check`、pre/post checker、OoO 前后图不变量对比。
- `ExpandFunction`、`AssignMemoryType`、`GraphPartition`、`GenerateMoveOp`、`SubgraphToFunction`、`InferParamIndex`、`AddAlloc`、`RemoveAlloc`、`CopyOutResolve`、`InsertSync`、`CodegenPreproc` 等结构性 Pass 默认不建议禁用；这些 Pass 被跳过后，下游依赖不会自动重验。
- `CommonOperationEliminate` / COE 是可临时禁用的优化 Pass，可用于本地隔离；若禁用后问题消失，继续定位 COE 的 skip opcode、hash、删边/改图逻辑。
- 若存在更窄的功能开关，优先建议功能开关，而不是关整个 Pass，例如 `pass.enable_vf=false`、`pass.auto_mix_partition=0`、`pass.copyout_resolve_coalescing=0`、`pass.ooo_sched_mode`。
- 当前 Python `PassConfigKey` 只暴露 `KEY_DUMP_GRAPH`；不要编造 `pypto.PassConfigKey.KEY_DISABLE_PASS` / `KEY_HEALTH_CHECK` 示例。非 dump 配置应指向 `tile_fwk_config.json` 或项目支持的配置路径。
- 对未知 Pass，只建议开启 dump/checker 或查配置与依赖，不主动建议 `disable_pass`。
