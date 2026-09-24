# 诊断动作可行性

“临时关闭某个 Pass 看现象是否消失”只能用于可禁用优化 Pass，不能作为通用建议。

| Pass / 模块 | 策略 | 可用诊断动作 | 禁止建议 |
|-------------|------|--------------|----------|
| `OoOSchedule` / `OoOScheduler` / OoO schedule | 必要调度 Pass | 开启 `dump_graph`、`health_check`、pre/post checker；比较 OoO 前后图不变量、memory range、spill 输出、event/sync 信息 | 不要建议关闭或绕过 OoO schedule |
| `ExpandFunction`、`AssignMemoryType`、`GraphPartition`、`GenerateMoveOp`、`SubgraphToFunction`、`InferParamIndex`、`AddAlloc`、`RemoveAlloc`、`CopyOutResolve`、`InsertSync`、`CodegenPreproc` 等 | 结构性 / 依赖敏感 Pass | dump/checker/health report；比较输入输出不变量；查 `PassDependency` 和下游消费关系 | 不要作为默认二分对象禁用 |
| `CommonOperationEliminate` / COE | 可临时禁用的优化 Pass | 本地复现时可临时 `disable_pass`，对比 COE 前后图；若禁用后现象消失，继续检查 skip opcode、hash、删边/改图逻辑 | 不要把“长期禁用 COE”作为最终修复 |
| VF / ReduceCopy auto-mix / CopyOut coalescing 等功能 | 优先用窄开关 | `pass.enable_vf=false`、`pass.auto_mix_partition=0`、`pass.copyout_resolve_coalescing=0`、`pass.ooo_sched_mode` | 不要先关整个 Pass |
| 未知 Pass | 未确认 | 先查 pass 配置和依赖关系，只建议 dump/checker | 不要主动建议 `disable_pass` |

配置提示：当前公开 Python `PassConfigKey` 只暴露 `KEY_DUMP_GRAPH`，不要编造 `pypto.PassConfigKey.KEY_DISABLE_PASS` / `KEY_HEALTH_CHECK` 示例；非 dump 配置应指向 `tile_fwk_config.json` 或项目支持的配置路径。

---
