---
name: pypto-skill-reviewer
description: 对一个 skill 目录进行质量与最佳实践合规性评审并评分，同时检查 references 文件与 docs 目录的一致性。用于需要审计某个 skill、检查 skill 是否遵循规范、或在发布前评估 skill 的场景。触发词：评审 skill、skill 质量、skill 审计、检查 references。
---

# PyPTO Skill Reviewer

基于 10 个维度下的 52 条规则，对一个 skill 目录进行评审，并产出带有可执行修复建议的评分报告。

## 输入

用户会提供一个 `<skill-path>` —— 待评审 skill 目录的路径。该目录必须包含 `SKILL.md` 文件。

## 参考文件

| File | Purpose | Load Timing |
|------|---------|-------------|
| [references/rules.json](references/rules.json) | 52 条规则、维度、权重和严重级别的唯一事实来源 | 在第 1 阶段和第 2 阶段开始时读取 |
| [references/scoring-spec.md](references/scoring-spec.md) | 评分算法：维度权重、扣分公式、S0 否决、等级映射 | 在第 3 阶段开始时读取 |
| [references/semantic-checklist.md](references/semantic-checklist.md) | 22 条语义规则的详细检查要求、证据标准和判定准则 | 在第 2 阶段开始时读取 |
| [references/knowledge-checklist.md](references/knowledge-checklist.md) | 知识库一致性检查清单：检查范围、内容正确性、语义一致性检查项 | 在第 4 阶段开始时读取 |
| [scripts/validate_skill.py](scripts/validate_skill.py) | 对 26 条静态规则进行确定性静态检查，输出 JSON findings | 在第 1 阶段执行 |
| [scripts/score_findings.py](scripts/score_findings.py) | 对合并后的 findings 执行确定性打分与覆盖率统计，输出 JSON score | 在第 3 阶段执行 |
| [templates/report-template.md](templates/report-template.md) | 最终评审报告的 Markdown 模板 | 在第 3 阶段开始时读取 |

## 工作流程

执行四个阶段。第 1 阶段和第 2 阶段可并行运行。第 4 阶段检查知识库一致性（references/ + SKILL.md 知识内容）。

### 第 1 阶段：静态检查

1. 读取 [references/rules.json](references/rules.json) 以理解规则定义，因为后续语义评审需识别哪些规则是 semantic 类型，知识检查需识别哪些规则是 knowledge 类型，且需从 rules.json 查询 rule_content 用于问题聚合。
2. 对目标 skill 运行静态检查器，因为 26 条静态规则可通过脚本确定性检查，避免人工误判：

   ```bash
   python3 scripts/validate_skill.py <skill-path>
   ```

3. 捕获 JSON 输出 —— 一个 finding 对象数组 `findings_static`。
4. 验证脚本输出的字段完整性：
   - 检查每个 finding 包含所有必需字段（rule_id/status/severity/dimension/message/evidence/suggested_fix）
   - 缺失字段应补充默认值：
     - PASS findings：suggested_fix = ""
     - FAIL findings：必须提供非空的 suggested_fix（若脚本未提供，需在后续步骤补充）
5. 验证脚本是否成功退出。若失败，报告错误，并仅继续输出第 2 阶段结果。

### 第 2 阶段：语义评审

1. 读取 [references/rules.json](references/rules.json) 以识别哪些规则是 `type: "semantic"`。
2. 读取 [references/semantic-checklist.md](references/semantic-checklist.md) 获取详细检查流程。
3. 读取目标 skill 目录中的全部文件：
   - `SKILL.md`（必需）
   - 子目录中的所有文件（`references/`、`scripts/`、`templates/` 等）
4. 严格按照清单流程，逐条评估语义规则与目标 skill 内容的一致性。
5. 对每条规则，生成包含必需字段的 finding 对象：
   ```json
   {
     "rule_id": "R07",
     "status": "失败|通过|跳过",
     "severity": "S1",
     "dimension": "D1",
     "message": "（必须引用目标 skill 的具体内容）",
     "evidence": {
       "file": "SKILL.md",
       "line": 3,
       "snippet": "（来自目标 skill 的逐字摘录，≥10 个字符）"
     },
     "suggested_fix": "（针对该具体 skill 的明确修改建议）"
   }
   ```
6. 对所有语义 findings 执行自校验：
   - **Snippet 匹配**：验证每个 `evidence.snippet` 都在目标 skill 文件中逐字存在。若发现伪造片段，删除或修正该 finding。
   - **唯一性**：确保任意两条 finding 的 `message` 文本不完全相同。对重复项进行合并或差异化处理。
   - **具体性**：确认每条 `message` 和 `suggested_fix` 都引用目标 skill 的具体内容，而非泛化建议。

### 第 3 阶段：评分与报告

1. 读取 [references/scoring-spec.md](references/scoring-spec.md) 获取评分算法。
2. 读取 [templates/report-template.md](templates/report-template.md) 获取报告格式。
3. 将第 1 阶段和第 2 阶段的所有 findings 合并为单一列表，并保存为 `findings_merged.json`。
4. 对合并后的 findings 运行确定性评分脚本：

   ```bash
   python3 scripts/score_findings.py \
     --rules references/rules.json \
     --skill-path <skill-path> \
     --findings findings_merged.json
   ```

   将 JSON 输出保存为 `score_result.json` 文件。
5. 按位置将 findings 聚合为问题：
   - **聚合键**：`file + line_range`（彼此相距 ±5 行内的 findings 合并为一个问题）
   - 每个问题记录所有匹配的 `rule_id` 值
   - 每个问题生成一条统一修复建议（包含修改前/后对比）
   - `rule_content`：从 rules.json 中查询对应 rule_id 的 `rule` 字段内容
6. 分数、等级、覆盖率、计数（pass/fail/warn/skip）必须使用 `score_result`，不得手工估算。
7. 按维度计算分数时，以 `score_result.dimensions` 为唯一来源：

   ```python
   dimension_raw   = max(0, 100 - sum_of_FAIL_deductions)
   dimension_score = dimension_raw × weight
   ```

8. 应用质量门禁 —— 评分前过滤 findings：
    - **内部错绑**：若某语义 finding 的证据明显引用的是 reviewer 自身而非目标 skill，移除该 finding，并标注原因 `internal_misbound_rule_or_evidence`。
    - **证据不足**：若某语义 finding 的 `evidence.snippet` 无法在目标文件中逐字找到，移除该 finding，并标注原因 `low_information_snippet`。
    - 为报告中的质量门禁章节记录所有被过滤项。
9. 使用模板渲染最终报告，填充全部占位符：
    - **禁止跳过执行步骤**：若生成辅助脚本（如 Python 脚本），必须先执行该脚本再读取结果文件
    - **执行顺序**：生成脚本 → 执行脚本 → 脚本输出结果文件 → 验证文件存在 → 读取文件内容
    - **错误处理**：若脚本执行失败，记录错误并跳过该步骤，不可陷入"生成脚本→尝试读取结果→失败→重试"的死循环
    - **直接渲染**：仅在无法使用辅助脚本时，才可直接使用模板渲染报告内容（不推荐）

### 第 4 阶段：知识库一致性检查

此阶段检查 skill 中的知识内容与 `docs/` 目录的一致性，处理 knowledge 类型规则（R49-R52）。

**检查范围**：
- `references/` 目录（若存在）
- `SKILL.md` 中的知识性内容（API 说明、路径说明、术语定义等）

**排除内容**：流程语义内容（如"运行脚本""读取文件"）不需要检查。

1. 读取 [references/rules.json](references/rules.json) 识别 knowledge 类型规则（R49-R52）。
2. 读取 [references/knowledge-checklist.md](references/knowledge-checklist.md) 获取详细检查流程。
3. 扫描 `references/` 目录（若存在），获取文件列表。
4. 提取 `SKILL.md` 知识性内容：
   - API：`pypto.xxx` API 名称及说明
   - 路径：文件路径、命令路径引用
   - 术语：框架概念术语（tile、tensor、pass、codegen 等）
5. 对提取的知识内容执行一致性检查（识别实体 → 搜索 docs 验证 → 对比分析）。
6. 问题分类（P0/P1/P2）并二次验证。
7. 生成 R49-R52 finding 对象，添加到合并列表。
8. 将知识库检查结果作为独立章节添加到评审报告。

## 输出

### 必需产出文件

评审完成后，必须产出以下文件：

| 文件名 | 说明 | 生成阶段 |
|--------|------|---------|
| `findings_merged.json` | 合并后的完整 findings（静态 + 语义） | 第 3 阶段步骤 3 |
| `score_result.json` | 评分脚本输出结果 | 第 3 阶段步骤 4 |

### Markdown 评审报告

直接向用户输出完整的 Markdown 评审报告。报告必须包含以下全部章节 —— 缺少任意章节都视为不完整。

**核心评审章节（7 个）**：

1. **评审摘要** —— skill 名称、总分（0-100，保留两位小数）、等级（A/B/C/D/F）、S0 否决状态（是/否）、规则统计（pass/fail/warn/skip 计数）
2. **维度评分表** —— 10 行（D1-D10），每行包含：原始分（0-100）、权重、加权分、扣分明细（列出每个 FAIL 的 rule_id 及其扣分值）
3. **规则覆盖率** —— 静态计数 + 语义计数 + 知识计数 = 总评估数，含按状态拆分（PASS/FAIL/SKIP），覆盖率百分比 = evaluated / 52 × 100%
4. **质量门禁** —— 被过滤项的数量与移除原因（internal_misbound / low_information_snippet）
5. **问题列表** —— 按严重级别分组（S0 → S3），每个问题包含：
   - 引用目标 skill 具体内容的问题描述
   - 带严重级别标记的匹配规则 ID
   - 位置（`file:line`）与证据（来自目标的逐字片段，≥10 个字符）
   - 含修改前/后对比的具体修复建议
6. **通过规则汇总** —— 按维度分组的全部通过 rule_id
7. **知识库一致性检查** —— 包含以下子章节：
   - 检查概况（references/ 文件数量、SKILL.md 知识内容、问题统计、R49-R52 规则状态）
   - P0 必须修复问题列表
   - P1 建议修复问题列表
   - P2 可选修复问题列表
   - 无需修复项说明（若有）
   - 联动修改检查（若有）

若 skill 中不存在知识性内容（既无 references/ 目录，SKILL.md 中也无 API/路径/术语等知识内容），在第 7 章节注明："未检测到知识性内容，R49-R52 自动 SKIP"。

**成功标准**：一个有效报告需满足：
  (a) 核心 7 个章节全部存在，
  (b) 总分 = 各维度加权分之和（±0.01），
  (c) 每条 finding 的 evidence.snippet 都在目标文件中逐字存在，
  (d) 覆盖率分母 = 52，
  (e) 第 7 章节（知识库一致性检查）必须存在。

## 错误处理

- **未找到 SKILL.md**：以单条 S0 finding（R01）上报，跳过其他所有检查，输出最小报告（分数 0，等级 F）。
- **脚本执行失败**：记录错误，仅继续语义评审，并在报告中注明静态检查未完成。覆盖率中 26 条静态规则全部标记为 SKIP。
- **脚本输出格式错误**：若 validate_skill.py 的 stdout 不是合法 JSON：
  1. 报告："静态分析脚本返回了非 JSON 输出"
  2. 在报告质量门禁章节包含原始 stderr（前 500 个字符）
  3. 继续仅语义评审
  4. 覆盖率中 26 条静态规则全部标记为 SKIP
- **skill 目录为空**：同“未找到 SKILL.md”。
- **frontmatter 无效**：R01 FAIL 触发 S0 否决。尽可能继续检查其他规则（即不依赖 frontmatter 数据的规则）。

## 约束

- 不要修改目标 skill 目录中的任何文件 —— 这是只读评审。
- 不要伪造证据 —— 每个 snippet 都必须真实存在于目标文件中。
- **`rules.json` 是 52 条规则的唯一事实来源**。禁止：
  - 发明 rules.json 未定义的规则
  - 修改 rules.json 中的严重级别或维度归属
  - 跳过任何规则 —— 若某规则无法检查，必须以 SKIP 标记并给出原因
  - 基于假设或外部知识覆盖规则定义
- 每条 finding 都必须引用一个存在于 rules.json 的 `rule_id`。
- 必须严格使用 `scoring-spec.md` 中的评分公式 —— 不得估算或近似。
- **脚本执行约束**：
  - 若生成辅助脚本（Python/Shell 等），必须立即执行该脚本
  - 禁止跳过执行步骤直接读取结果文件（会导致文件不存在错误）
  - 执行脚本失败时，记录错误并继续，不可陷入死循环
  - 正确流程：Write script → Bash execute script → Read result file
- **Python 代码生成约束**（用于生成临时脚本时）：
  - 所有 Python 变量名、字符串内容必须使用纯 ASCII 字符
  - 禁止在 Python 代码中使用中文标点符号（如 `、` `。` `：` 等）
  - 注释和说明性文本可以使用中文，但必须使用标准 ASCII 标点（逗号、句号、冒号）
  - 字符串值若需包含中文文本，必须使用英文标点符号分隔
- **知识规则检查的额外约束**：
  - 以 `docs/` 目录为唯一标杆，不以经验推断代替文档验证
  - 标记问题前必须确认执行上下文（工作目录、环境变量、前置条件）
  - 合理简化、更严格规范、内部知识不属于问题，不应误报
