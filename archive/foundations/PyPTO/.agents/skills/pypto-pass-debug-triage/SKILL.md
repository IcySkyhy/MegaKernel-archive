---
name: pypto-pass-debug-triage
description: PyPTO 跨模块问题分诊与历史 bug 模式匹配技能。仅在故障归属尚不明确、需要先判断是否应归因于 Pass，或需要匹配随包静态历史模式时使用；输出可验证的根因假设和排查方向，不替代已确认 Pass 报错后的源码定位、模块分析、业务流分析、性能优化或 UT 生成技能。
---

# PyPTO Pass Debug Triage

本技能是 Pass 问题的分诊入口。先判断根因是否真在 Pass，再用本地静态模式库生成可验证的修复假设。不要把模式匹配结果说成已定位事实；除非已经读取当前源码或用户提供了足够证据，否则统一表述为“假设/优先排查方向”。

## 核心原则

- 先做跨模块定界，再做 Pass 内模式匹配。
- 以 `references/patterns/` 下的分类 JSON 为结构化唯一事实源，Markdown reference 只补充导航和解释；修改 JSON 时必须同步核对对应 Catalog 索引。
- 输出要能落地到文件、函数、反模式、修复方向和验证动作。
- 所有“关掉某个 Pass 试试”的建议必须先过 `diagnostic_action_policy`；必要或结构性 Pass 只能建议开 dump/checker/health report、前后图对比或更窄的 feature flag，不能建议禁用。
- 本技能默认只做 triage 和修复建议；用户明确要求继续改代码、跑测试或生成 UT 时，再切到对应工作流或建议使用相关 skill。

## 与其他 Pass skill 的边界

| 用户目标 | 使用 skill | 边界 |
|----------|------------|------|
| 故障归属未知，先判断 Pass / operation / machine / IR 等模块，或匹配历史反模式 | `pypto-pass-debug-triage` | 本技能只输出候选假设、证据缺口和验证动作 |
| 日志或调用栈已经确认是 Pass 报错，需要读取当前源码定位错误 | `pypto-pass-error-locator` | 不再重复做全模块分诊 |
| 理解某个 Pass 的接口、功能和实现 | `pypto-pass-module-analyzer` | 输出模块分析文档，不做故障模式匹配 |
| 理解多个 Pass 的执行顺序、依赖和数据流 | `pypto-pass-workflow-analyzer` | 面向业务流程，不做单点故障修复 |
| 优化 Pass 编译耗时 | `pypto-pass-perf-optimizer` | 仅处理性能问题 |
| 根据业务描述生成 Pass UT | `pypto-pass-ut-generate` | 仅生成测试，不承担故障定界 |

## 输入抽取

从用户输入中提取这些字段，缺失字段允许为空：

| 字段 | 说明 |
|------|------|
| `pass_name` | 用户提到的 Pass 名，或从日志/路径推断 |
| `symptom` | 报错文本、精度异常、图结构异常、运行时现象 |
| `files` | 用户给出的路径，或日志中的文件路径 |
| `scenario` | 关键词，如 spill、reshape、validshape、overlap、L0C、OoO |
| `evidence_level` | 日志/源码/仅口述，用来控制置信度 |

若用户只给出很泛的描述，也要先给出需要补充的证据清单，同时基于已有关键词做低置信度初筛。

## Reference 加载规则

只读取当前任务需要的 reference，避免一次性加载整个知识库。

| 需要解决的问题 | 读取文件 |
|----------------|----------|
| 所有 triage 任务 | `references/patterns/meta.json`、`references/patterns/disable-policy.json` |
| 跨模块定界、交接话术、常见误判 | `references/patterns/cross-module.json`、`references/triage-protocol.md` |
| Pass 内历史修复模式 | `references/patterns/pass-patterns.json` |
| Pass 源码审计模式 | `references/patterns/source-patterns.json` |
| 非 Pass 模块模式 | `references/patterns/non-pass-patterns.json` |
| 解释 P/S/N 模式细节、查人类可读导航 | `references/catalog/index.md` 及同名分类文件 |
| 用户提到具体 Pass 或准备改 Pass | 先查 `references/pass-memory/index.md`，再读取其指向的对应文件和词条 |
| 新增 Pass 设计记忆词条 | `templates/pass-memory-entry.md` |
| 审查已有 diff 或修改前风险评估 | `references/change-safety-check.md` |
| 新增 Pass 功能、重构 Pass 或做 code review | `references/pass-change-review-guide.md` |

读取 Pass 设计记忆时，先在 `references/pass-memory/index.md` 查找 Pass 名或别名，再到索引指向的文件搜索 `## {PassName}`；找不到精确章节时，按“查找提示/别名”定位内部组件章节，例如 `OoOSchedule` 对应 `OoOScheduler` / `SpillBuffer`，`GraphPartition` 对应 `SupernodeGraphBuilder` / `OspPartitioner`。

## 工作流

### 1. 解析症状

归一化关键词：

- 将日志中的路径、函数名、Pass 名、opcode、dtype、memory type、错误码加入匹配词。
- 将中文现象映射到英文关键词，例如“精度不对”映射为 `precision` / `accuracy`，“越界”映射为 `out_of_range` / `overflow` / `memory overlap`。
- 保留原始症状，报告中必须能追溯到用户提供的证据。

### 2. 跨模块定界

读取 `references/patterns/cross-module.json.cross_module_triage`，按 `symptom_keywords` 匹配候选模块。若命中多个模块，按以下优先级裁剪：

1. 直接报错路径所在模块。
2. 症状关键词命中数量。
3. `priority` 字段。
4. 用户给出的 Pass 名或文件路径。

若最可能模块不是 Pass，仍可保留 Pass 相关假设，但报告第 2 节必须先明确非 Pass 排查方向。

### 3. 模式匹配

Pass 内匹配：

- 分别读取 `references/patterns/pass-patterns.json.patterns` 和 `references/patterns/source-patterns.json.source_patterns`。
- `pass_modules` 命中：+3。
- `phenomenon_keywords` 命中：每个 +2。
- `files` 命中：+2。
- `severity=high` 且症状直接相关：+1。

非 Pass 匹配：

- 读取 `references/patterns/non-pass-patterns.json.non_pass_source_patterns`。
- `likely_modules` 命中：+3。
- `symptom_keywords` 命中：每个 +2。
- 若条目没有 `symptom_keywords`，用 `pass_symptoms` 作为症状匹配字段。
- `files_to_check` 命中：+2。

输出 Top 3，允许混合 Pxxx、Sxxx、Nxxx。若分数接近，优先展示能直接解释用户症状且有明确验证动作的模式。

### 4. 生成修复假设

每个候选模式必须包含：

- 根因假设：一句话。
- 定界依据：命中的关键词、路径或 Pass 名。
- 代码反模式：来自 `code_anti_pattern`。
- 修复方向：来自 `fix_pattern`，必要时结合当前源码路径描述。
- 验证动作：来自 `verification_hints`，并补充应 dump 的中间状态。
- 置信度：高/中/低。没有日志或源码证据时不得给“高”。

### 5. 建议动作可行性检查

生成“下一步”之前，读取 `references/patterns/disable-policy.json.diagnostic_action_policy` 并过滤不可执行建议：

- 若怀疑对象命中 `required_passes` 或 `structural_or_dependency_sensitive_passes`，不要建议关闭或绕过它；改为建议开启 dump/checker/health report、比较前后图不变量，或使用更窄的 feature flag。
- 若建议改 pass 配置，先看 `config_api_caveat`：当前 Python `PassConfigKey` 只暴露 `KEY_DUMP_GRAPH`，不要编造 `KEY_DISABLE_PASS` / `KEY_HEALTH_CHECK` 的 Python 示例。
- `CommonOperationEliminate` / COE 属于可临时禁用的优化 Pass，可作为本地隔离实验；若禁用后现象消失，最终修复仍应落到 COE 的 skip opcode、hash、删边/改图逻辑上。
- 对策略表未知的 Pass，不要主动建议 `disable_pass`；除非当前源码/配置或用户明确说明该 Pass 可安全禁用。
- 不要把“禁用 Pass 后规避问题”写成最终修复方案，只能写成本地隔离验证手段。

### 6. 输出报告

使用下面格式，保留高信号内容，避免把完整模式库粘贴给用户。

```markdown
# PyPTO Bug 定界与修复建议报告

## 1. 症状摘要
- 报错位置/Pass：{pass_name}
- 现象：{symptom}
- 相关文件：{files}
- 业务场景：{scenario}

## 2. 跨模块定界
- 最可能模块：{module}
- 定界依据：{matched_keywords_or_paths}
- 是否继续 Pass 内分析：{是/否，以及原因}
- 优先排查文件：{files_to_check}

## 3. 候选根因
### 假设 1：{pattern_id} - {category_cn}（置信度：{高/中/低}）
- 根因假设：{root_cause}
- 代码反模式：{code_anti_pattern}
- 修复方向：{fix_pattern}
- 建议查看：{file/function}
- 验证动作：{verification_hints}

## 4. 推荐优先级
1. {pattern_id}：{reason}
2. {pattern_id}：{reason}

## 5. 下一步
- 建议动作约束：{哪些 Pass 不能关，哪些可临时禁用，以及原因}
- {最小验证动作}
- {需要补充的日志/源码/dump}
```

## 边界与禁止事项

- 不要编造不存在的 commit、文件路径、函数名或 issue。
- 不要在没有源码/日志证据时说“已经定位”或“已经修复”。
- 不要把非 Pass 模块问题硬归因到 Pass；跨模块定界结论必须优先展示。
- 不要建议禁用必要或结构性 Pass（尤其是 `OoOSchedule` / `OoOScheduler`、`AddAlloc`、`InsertSync`、`CodegenPreproc` 等）；只能建议开启 dump、checker、health report、前后图对比或更窄的 feature flag。
- 不要在线检索 git 历史来补知识库；本技能只依赖本地仓库和随包 reference。
- 如果 reference 中路径不存在，先用本地 `rg --files` 查当前真实路径，并在报告中说明“知识库路径已迁移/需更新”。
