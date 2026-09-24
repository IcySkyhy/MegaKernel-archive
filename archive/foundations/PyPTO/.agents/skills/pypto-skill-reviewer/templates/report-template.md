# 技能评审报告

## 评审摘要

| 项目 | 结果 |
|------|------|
| 技能名称 | {{skill_name}} |
| 评审时间 | {{timestamp}} |
| 总分 | {{score}} / 100 |
| 等级 | {{grade}} |
| S0 否决 | {{s0_veto}} |
| 规则统计 | 通过 {{pass_count}} / 失败 {{fail_count}} / 警告 {{warn_count}} / 跳过 {{skip_count}} |

## 维度得分

| 维度 | 名称 | 权重 | 满分 | 得分 | 扣分详情 |
|------|------|------|------|------|---------|
| D1 | Frontmatter 元数据 | 25% | 25.0 | {{d1_score}} | {{d1_detail}} |
| D2 | 简洁性与效率 | 15% | 15.0 | {{d2_score}} | {{d2_detail}} |
| D3 | 文件结构与导航 | 5% | 5.0 | {{d3_score}} | {{d3_detail}} |
| D4 | 语言与表达 | 10% | 10.0 | {{d4_score}} | {{d4_detail}} |
| D5 | 精确性与可执行性 | 10% | 10.0 | {{d5_score}} | {{d5_detail}} |
| D6 | 工作流完整性 | 10% | 10.0 | {{d6_score}} | {{d6_detail}} |
| D7 | 模式与最佳实践 | 5% | 5.0 | {{d7_score}} | {{d7_detail}} |
| D8 | 反模式检测 | 10% | 10.0 | {{d8_score}} | {{d8_detail}} |
| D9 | 脚本与代码质量 | 5% | 5.0 | {{d9_score}} | {{d9_detail}} |
| D10 | References 一致性 | 5% | 5.0 | {{d10_score}} | {{d10_detail}} |

## 规则覆盖率

| 指标 | 数值 |
|------|------|
| 期望规则数 | {{expected_rules}} |
| 已评估规则数 | {{evaluated_rules}} |
| 跳过规则数 | {{skipped_rules}} |
| 覆盖率 | {{coverage_pct}}% |

{{#if skipped_rule_list}}
**跳过的规则**：{{skipped_rule_list}}（原因：不适用于此技能）
{{/if}}

### 规则状态明细

| 规则 | 状态 | 维度 | 严重度 | 类型 |
|------|------|------|--------|------|
{{#each all_rules}}
| {{rule_id}} | {{status}} | {{dimension}} | {{severity}} | {{type}} |
{{/each}}

## 质量门禁

| 类型 | 数量 | 说明 |
|------|------|------|
| 内部误绑条目 | {{misbound_count}} | 发现与目标技能无关，已从评分中剔除 |
| 证据不足条目 | {{insufficient_count}} | snippet 无法在源文件中匹配，已从评分中剔除 |

{{#if gate_filtered_items}}
**被过滤的条目**：
{{#each gate_filtered_items}}
- {{rule_id}}：{{reason}}
{{/each}}
{{/if}}

## 问题清单

### S0 致命缺陷

{{#each s0_issues}}
#### 问题 {{n}}：{{issue_title}}

**命中规则**：{{rule_ids_with_severity}}

> 规则内容：{{rule_content}}

**位置**：`{{file}}:{{line}}`

**当前内容**：
> {{current_content}}

**问题说明**：
{{explanation}}

**修改建议**：
> {{suggested_fix}}

---
{{/each}}

### S1 重大问题

{{#each s1_issues}}
#### 问题 {{n}}：{{issue_title}}

**命中规则**：{{rule_ids_with_severity}}

> 规则内容：{{rule_content}}

**位置**：`{{file}}:{{line}}`

**当前内容**：
> {{current_content}}

**问题说明**：
{{explanation}}

**修改建议**：
> {{suggested_fix}}

---
{{/each}}

### S2 中等问题

{{#each s2_issues}}
#### 问题 {{n}}：{{issue_title}}

**命中规则**：{{rule_ids_with_severity}}

> 规则内容：{{rule_content}}

**位置**：`{{file}}:{{line}}`

**当前内容**：
> {{current_content}}

**问题说明**：
{{explanation}}

**修改建议**：
> {{suggested_fix}}

---
{{/each}}

### S3 轻微建议

{{#each s3_issues}}
#### 问题 {{n}}：{{issue_title}}

**命中规则**：{{rule_ids_with_severity}}

> 规则内容：{{rule_content}}

**位置**：`{{file}}:{{line}}`

**当前内容**：
> {{current_content}}

**问题说明**：
{{explanation}}

**修改建议**：
> {{suggested_fix}}

---
{{/each}}

## 通过项

共 {{pass_count}} 条规则通过。

| 维度 | 通过规则 |
|------|---------|
{{#each dimension_passes}}
| {{dimension}} | {{passed_rules}} |
{{/each}}

{{#if has_references}}

## 知识库一致性检查

### 检查概况

| 统计项 | 数量 |
|--------|------|
| references/ 文件数 | {{ref_file_count}} |
| SKILL.md 知识内容 | {{skill_knowledge_count}} |
| P0 问题数 | {{ref_p0_count}} |
| P1 问题数 | {{ref_p1_count}} |
| P2 问题数 | {{ref_p2_count}} |

{{#if ref_p0_issues}}

### P0 必须修复

> 事实性错误，必须修复

{{#each ref_p0_issues}}

#### 问题 {{n}}：{{issue_title}}

- **文件**：`{{file}}:{{line}}`
- **问题**：{{problem}}
- **docs 对比**：{{docs_comparison}}
- **修复建议**：{{fix_suggestion}}
- **联动修改**：{{#if linkage}}{{linkage}}{{else}}无{{/if}}

---
{{/each}}
{{/if}}

{{#if ref_p1_issues}}

### P1 建议修复

> 歧义或可能导致误解的问题，建议修复

{{#each ref_p1_issues}}

#### 问题 {{n}}：{{issue_title}}

- **文件**：`{{file}}:{{line}}`
- **问题**：{{problem}}
- **docs 对比**：{{docs_comparison}}
- **修复建议**：{{fix_suggestion}}
- **联动修改**：{{#if linkage}}{{linkage}}{{else}}无{{/if}}

---
{{/each}}
{{/if}}

{{#if ref_p2_issues}}

### P2 可选修复

> 模糊或不够清晰的问题，可选修复

{{#each ref_p2_issues}}

#### 问题 {{n}}：{{issue_title}}

- **文件**：`{{file}}:{{line}}`
- **问题**：{{problem}}
- **docs 对比**：{{docs_comparison}}
- **修复建议**：{{fix_suggestion}}
- **联动修改**：{{#if linkage}}{{linkage}}{{else}}无{{/if}}

---
{{/each}}
{{/if}}

{{#if ref_no_fix_items}}

### 无需修复项说明

{{#each ref_no_fix_items}}
- {{item}}：{{reason}}
{{/each}}
{{/if}}

{{#if ref_linkage_check}}

### 联动修改检查

{{ref_linkage_check}}
{{/if}}
{{else}}
**知识库一致性检查**：未检测到知识性内容，R49-R52 自动 SKIP。
{{/if}}
