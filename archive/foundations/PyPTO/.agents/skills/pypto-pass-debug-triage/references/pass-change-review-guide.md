# Pass 新功能与 Review 指南

## 使用时机

开发新 Pass、重构已有 Pass、或 review 涉及 Pass 输出不变量的 PR 时使用本文件。本文件把 `patterns/` 下的分类 JSON 作为防御清单使用，而不是只在出问题后定位。

## 设计阶段

在写代码前先回答：

| 检查项 | 对应模式 | 设计决策 |
|--------|---------|---------|
| 新功能是否涉及 spill/L0C/L1/dtype 路径？ | C001 | 需要和 machine 模块的同步、内存分配对齐 |
| 是否涉及 reshape/view/assemble/validshape？ | C002 | 明确 shape 推导来源是 operation、frontend 还是 Pass 自身 |
| 是否新增或修改具体算子行为？ | C003 | 先在 operation 层确认算子约束，再考虑 Pass 特判 |
| 是否涉及 runtime/launch/sync/memory pool？ | C004 | 优先在 machine 层保证正确，Pass 只做合法假设 |
| 是否涉及 Python/jit/frontend/symbolic？ | C005 | 前端 shape/attr 必须正确传到 IR |
| 是否涉及 IR/token/dce/transform？ | C005 | SSA/token 转换必须可处理新结构 |
| 是否涉及 distributed/shmem/allreduce？ | C006 | 通信原语的 validshape/size 必须显式传递 |
| 是否涉及 codegen/tile/cce？ | C007 | Pass 输出的 tile shape/memory type 必须在 codegen 支持范围内 |

原则：不要在 Pass 里补偿上游模块缺陷。若新功能需要 Pass 特殊处理，先确认上游是否能给出正确输入。

## 编码阶段

逐条映射到源码模式：

- 容器与指针安全：S001-S003。
- 视图类 OP 一致性：S004。
- 内存类型与 dtype 路径：S005、S006、P006、P008、P015。
- 动态 shape 与 validshape：S009、P005、P016、P017。
- 统计、累加与下溢：S010、P003。
- 返回值与边界检查：S011、S012、P001、P022。
- 哈希、顺序与属性身份：S007、S013、P013、P019、P020。
- 图关系同步与子图边界：S017、P010、P011、P021。
- 硬编码常量与同步事件：S008、P014、P023。
- Pass 配置与禁用策略：S045。
- Axis-combine 与 shape-transform：P018。
- 特殊 copy / raw tensor 属性：P024、P025。

## 自审阶段

1. 根据 diff 修改文件匹配 `patterns/source-patterns.json.source_patterns` 和 `patterns/non-pass-patterns.json.non_pass_source_patterns`。
2. 对每个命中模式，在新代码中查找 `code_anti_pattern`。
3. 按 `patterns/disable-policy.json.diagnostic_action_policy` 检查建议动作：必要 Pass 不允许建议关闭，只有可禁用优化 Pass 可作为本地隔离实验。
4. 输出 Self-Review Report：
   - 哪些模式可能适用。
   - 当前代码是否存在对应反模式。
   - 建议的修复或验证动作。

## UT 阶段

| 模式 | UT 方向 |
|------|---------|
| S001 | 空 producers/consumers/operands |
| S002 | 多生产者/多消费者 tensor |
| S003 | 属性类型不匹配 |
| S004 | reshape/view/assemble 组合 |
| S005 | 混合内存类型消费者 |
| S006 | 大 shape 乘法溢出 |
| S007 | 同构图多次运行 hash 一致 |
| S008 | 不同平台阈值边界 |
| S009 | 动态 validshape/offset |
| S010 | 多次清理/删除路径 |
| S011 | helper 返回 FAILED/-1 |
| S012 | `i == size` 边界 |
| S013 | 拆分/克隆后属性索引 |
| S014 | 指针有序容器多次运行一致 |
| S015 | 函数名与语义一致 |
| S016 | 多 ASSEMBLE 消费者格式冲突 |
| S017 | 多 consumer 替换后图不变量 |
| P016 | 插入中间 tensor 后 DynValidShape 保持 |
| P017 | reshape->assemble 重排 validShape 小于 full tile |
| P018 | combineAxis 下 VIEW/ASSEMBLE/RESHAPE pad 轴一致 |
| P019 | COE 不消除 OP_VIEW / OP_VEC_DUP 等 skip opcode |
| P020 | clone 后 rawMagic fallback 查找原始 operand attr |
| P021 | ReduceCopy 合并不暴露内部 tensor 给外部子图 |
| P022 | 空 shape / -1 动态维 / 极大维默认 tile 推导 |
| P023 | InsertSync event-id 耗尽和 setPipe 迭代修改 |
| P024 | SplitRaw 后 OP_SHMEM_GET toOffset 更新 |
| P025 | COPY_OUT -> HUB 复用后 raw_shape 属性更新 |
| S045 | 禁用必要/结构性 Pass 后下游依赖不变量仍被检查 |

## Review 提问

- 这个功能是否让 Pass 处理原本不支持的 opcode/dtype/format？operation 层是否已经支持？
- 新增循环/索引是否有边界保护？
- 是否可能出现空容器解引用？
- 是否涉及多生产者/多消费者？是否只处理第一个？
- 新增内存类型/dtype 推断是否覆盖所有合法组合？
- 动态 shape 场景是否被显式拒绝或正确处理？
- 是否引入硬编码常量？
- 是否检查 helper 返回值？
- Review 建议里是否把必要 Pass（如 `OoOSchedule` / `OoOScheduler`）错误写成可禁用？若需要隔离优化影响，是否只针对 COE 等可禁用优化 Pass？
- 是否编造了 Python `PassConfigKey.KEY_DISABLE_PASS` / `KEY_HEALTH_CHECK` 示例？当前公开 Python enum 只暴露 `KEY_DUMP_GRAPH`。
