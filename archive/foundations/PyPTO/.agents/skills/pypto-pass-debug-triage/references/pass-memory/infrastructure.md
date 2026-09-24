# Infrastructure

## PassManager

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | PassManager |
| 所属目录 | `framework/src/passes/pass_mgr/` |
| 主要源文件 | `pass_manager.cpp`、`pass_manager.h` |
| Pipeline 阶段 | 全阶段（编排 tensor/tile/block graph pass） |
| 前置依赖 Pass | 无（本模块是编排器） |
| 后置消费 Pass | 所有具体 Pass 均由此调度 |
| 对应 bug 模式 | S027、S045 |

### 2. 设计目标

管理 Pass 注册表、策略（strategy）与执行顺序；按策略名称取出 Pass 列表，逐个实例化、配置、执行，并在 `CFG_DEBUG_ALL` 或 `KEY_ENABLE_PASS_VERIFY` 开启时触发 dump 与校验。

### 3. 核心不变量

- 同一 strategy 内 Pass 的 `identifier` 不能重复。
- `PassDependency::Instance().CheckStrategyDependency(strategy, passes)` 必须在注册策略时通过。
- 运行时配置不能让必要/结构性 Pass 被静默跳过；若允许 `disablePass`，必须由策略或诊断规范限制（S045）。
- `RunPass` 的循环起始索引 `startIdx` 必须按 strategy 独立维护，不能被 `GetResumePath` 的副作用污染（S027）。
- 每个 Pass 执行后必须调用 `Program::GetInstance().VerifyPass(&function, i, identifier)`（当校验开关开启）。
- `ResetAllPasses` 必须保证每个 Pass 只被 reset 一次。

### 4. 关键设计决策与取舍

| 决策 | 选择 | 原因 | 如果改这里要注意什么 |
|------|------|------|---------------------|
| 单例模式 | `PassManager::Instance()` | 全局统一维护策略与注册表 | 多线程调用 `RegisterStrategy` 需加锁 |
| startIdx 成员变量 | `GetResumePath` 直接修改 `startIdx` | 支持从 resume path 恢复执行 | 切换 strategy 时必须重置，否则会导致跳过前置 Pass（S027） |
| 策略配置 | `pass_configs.json` + `ConfigManager` 按 `strategy+identifier` 取配置 | 支持 per-Pass 的 dump/verify/resume 开关 | 修改配置 key 时要同步 `PassConfigsDebugInfo` |
| Pass 禁用 | `Pass::Run` 看到 `disablePass` 直接返回 SUCCESS | 支持本地关闭可选优化 Pass | 对结构性 Pass 需额外保护；否则注册时依赖检查仍通过，但运行时缺少输出不变量 |
| 编译阶段终止 | `ShouldTerminateAtStage` 在 `ExpandFunction` / `SubgraphToFunction` 后终止 | 支持分阶段调试 | 新增终止点需更新 `kPassToStageMap` |
| Pass 生命周期 | `PassRegistry::CreatePass` 每次新建实例 | 保证不同 function/strategy 之间状态隔离 | 不要在 Pass 实现中依赖全局可变状态 |

### 5. 已知脆弱场景

- **模式 S027**：`startIdx` 是成员变量，`GetResumePath` 被某策略修改后，`RunPass` 执行另一策略会从中间 Pass 开始。触发条件：多策略流水线运行并触发 resume。检查点：`pass_manager.cpp:265-279` 的 `GetResumePath` 与 `RunPass` 的 `for (size_t i = startIdx; ...)`。典型报错：策略 B 运行时跳过 `InferTensorFormat` / `RemoveRedundantReshape` 等前置 Pass，导致后续 Pass 输入不变量不满足。
- **模式 S045**：`disablePass` 发生在单个 Pass 的 `Run` 内，`PassDependency` 不会感知某个已注册 Pass 在运行时被跳过。触发条件：建议关闭 `OoOSchedule`、`AddAlloc`、`InsertSync`、`CodegenPreproc` 等结构性 Pass 做二分。检查点：`Pass::Run` 的 `Pass [%s] is skipped.` 分支和 `diagnostic_action_policy`。

### 6. 常见被误判为 Pass 问题的症状

| 症状 | 实际常见根因模块 | 快速验证方法 |
|------|------------------|--------------|
| 某个 Pass 未被执行 | `pass_configs.json` 中该 Pass 被禁用，或 `startIdx` 被污染（S027） | 检查日志中 Pass 序号是否从 0 开始；检查 resume path 配置 |
| Pass 执行顺序错误 | `PassDependency` 校验未覆盖全部依赖，或策略列表被动态修改 | 检查 `BuildPvc2OooPassEntries` 中的顺序 |
| 所有 Pass 都执行但结果仍错 | 不是 PassManager 问题，而是某个具体 Pass 逻辑错误 | 开启 per-pass dump 定位 |
| `VerifyPass` 失败 | 具体 Pass 破坏不变量，而非编排器 | 查看 `VerifyPass` 报错的 identifier |
| 性能问题（Pass 反复执行） | 某个 Pass 内部循环，不是 PassManager 调度问题 | 查看每 Pass 的耗时日志 |

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`。

---

### 8. 调试快速入口

- 开启 Pass dump 的命令/环境变量：`CFG_COMPILE_DEBUG_MODE=CFG_DEBUG_ALL`（自动开启所有 Pass 的 printGraph/dumpGraph/printProgram）；`KEY_ENABLE_PASS_VERIFY=true`
- 关键中间状态：`strategies_`、`startIdx`、`strategyLogIndices_`、每个 Pass 的 `passDfxconfigs_`
- 推荐先跑的 UT：`framework/tests/.../pass_mgr/*`
- 相关 checker：无专属 checker；依赖各具体 Pass 的 checker

---

## Pass base class (pass_interface)

### 1. 基本信息

| 字段 | 内容 |
|------|------|
| Pass 名称 | `Pass` 基类（位于 `pass_interface/pass.{h,cpp}`；旧称 PassInterface） |
| 所属目录 | `framework/src/passes/pass_interface/` |
| 主要源文件 | `pass.cpp`、`pass.h` |
| Pipeline 阶段 | 全阶段（所有 Pass 的基类） |
| 前置依赖 Pass | 无 |
| 后置消费 Pass | 所有具体 Pass 继承实现 |
| 对应 bug 模式 | S026 |

### 2. 设计目标

定义所有 Pass 的公共生命周期：构造、运行（PreRun / RunOnFunction / PostRun）、dump/print、健康检查；由 `PassManager` 统一调度，由具体 Pass 重写业务逻辑。

### 3. 核心不变量

- `Run` 必须按顺序调用 `PreRun`、`RunOnFunction`、`PostRun`，任一失败则整体返回 FAILED。
- `disablePass` 只适合已确认可禁用的优化 Pass；必要/结构性 Pass 被跳过后，下游依赖不会自动重验（S045）。
- `PrintFunction` / `DumpFunctionJson` 遍历 `function.rootFunc_->programs_` 时，`subProgram.second` 可能为空必须跳过（S026）。
- `CreateLogFolder` 失败时 `passFolder_` 必须回退到合法路径，不能悬空。
- `PreCheck` / `PostCheck` 默认返回 SUCCESS；具体 Pass 可选择重写。
- `DeleteGraphFolderIfEmpty` 必须在成功或失败路径都调用，防止空目录残留。

### 4. 关键设计决策与取舍

| 决策 | 选择 | 原因 | 如果改这里要注意什么 |
|------|------|------|---------------------|
| 虚函数接口 | `RunOnFunction`、`PreCheck`、`PostCheck`、`DoHealthCheckBefore/After` 均为虚函数 | 具体 Pass 按需重写 | 新增 hook 时要保证默认实现不破坏已有 Pass |
| 统一 dump 命名 | `GetDumpFilePrefix` 生成 `Before/After_<idx>_<identifier>_<func>` 前缀 | 保证跨 Pass 文件名有序 | 修改命名规则要同步日志分析脚本 |
| 日志目录创建 | `CreateLogFolder` 按 strategy + Pass 序号创建目录 | 支持多 strategy 多次运行 | 并发运行同一 strategy 时需避免目录冲突 |
| 特殊 Pass dump 标记 | `ExpandFunction`、`RemoveRedundantReshape`、`SubgraphToFunction`、`CodegenPreproc` 有特殊 stage dump | 标记 tensor/tile/block graph 阶段边界 | 新增特殊 dump 点要更新 `handlePreRunDumpGraph` / `PostRun` |

### 5. 已知脆弱场景

- **模式 S026**：`Pass::DumpJsonFile` / `PrintFunction` / `DumpFunctionJson` 循环遍历 `function.rootFunc_->programs_` 时只检查 `rootFunc_`，未检查 `subProgram.second` 是否为空。触发条件：开启 `dumpGraph`/`printGraph`，且 leaf function 为空的异常图。检查点：`pass.cpp:162`、`pass.cpp:195`、`pass.cpp:211`。典型报错：空指针解引用导致 coredump。
- **模式 S045**：`Pass::Run` 在 `disablePass=true` 时直接跳过 `PreRun` / `RunOnFunction` / `PostRun` 并返回 SUCCESS；`PassDependency` 不会因运行时配置禁用而重新检查依赖。触发条件：将 `OoOSchedule`、`AddAlloc`、`InsertSync`、`CodegenPreproc` 等结构性 Pass 当作可关闭调试开关。典型后果：下游拿到缺失的不变量，错误表现偏离真正根因。

### 6. 常见被误判为 Pass 问题的症状

| 症状 | 实际常见根因模块 | 快速验证方法 |
|------|------------------|--------------|
| Pass 运行前 coredump | `function.rootFunc_` 或 `programs_` 构造不完整，问题在 `SubgraphToFunction` 或 `Program` | 检查 dump 堆栈是否停在 `pass.cpp` 的空指针解引用 |
| dump 文件缺失 | `config::LogTopFolder()` 未配置或目录创建失败 | 检查 `CreateLogFolder` 返回与文件系统权限 |
| PreCheck/PostCheck 失败 | 具体 Pass 重写的检查逻辑，不是基类问题 | 查看日志中具体失败的 identifier |
| 健康检查误报 | `DoHealthCheckBefore/After` 由具体 Pass 实现 | 关闭 `healthCheck` 后复现 |

### 7. 修改前安全检查清单

通用检查项统一见 `../change-safety-check.md`。

本词条补充检查项：

- [ ] 是否新增或建议了 `disable_pass`？若目标是必要/结构性 Pass，是否改成 dump/checker/health report 或更窄 feature flag？

---

### 8. 调试快速入口

- 开启 Pass dump 的命令/环境变量：`CFG_COMPILE_DEBUG_MODE=CFG_DEBUG_ALL`；或单独设置某 Pass 的 `printGraph=true`、`dumpGraph=true`
- 关键中间状态：`passFolder_`、`graphFolder_`、`identifier_`、`strategy_`、`passRuntimeIndex_`、`passDfxconfigs_`
- 推荐先跑的 UT：所有具体 Pass 的 UT 都会间接覆盖基类；构造 leaf function 为空的 case 可验证 S026
- 相关 checker：无专属 checker；依赖各具体 Pass 的 checker

---
