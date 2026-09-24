# 知识库一致性检查清单

本检查清单定义了对 skill 中的知识内容进行一致性检查的详细流程和标准，对应 R49-R52 四条知识规则。

**检查范围**：
- `references/` 目录（若存在）
- `SKILL.md` 中的知识性内容（API 说明、路径说明、术语定义等）

**排除内容**：流程语义内容（如"运行脚本""读取文件""检查结果"）不需要检查。

## 关联规则

| 规则 | 严重度 | 检查内容 |
|------|--------|----------|
| R49 | S1 | skill 中的知识内容（references/ 和 SKILL.md）应与 docs/zh/ 保持一致性，不得存在 P0 级别问题 |
| R50 | S2 | skill 中的路径应在正确的执行上下文中可访问 |
| R51 | S2 | skill 中引用的 API 应存在于 docs/zh/ 或官方示例中 |
| R52 | S2 | skill 中的术语应与 docs/zh/tutorials/appendix/glossary.md 保持一致 |

## 输出格式

对每个发现的问题，输出一个结构化对象：

```json
{
  "level": "P0",
  "type": "代码错误",
  "file": "references/example.md",
  "line": 10,
  "snippet": "（问题代码或描述的原文片段）",
  "description": "（具体问题描述）",
  "evidence": "（在 docs/zh/ 中找到的矛盾证据或验证命令结果）",
  "suggested_fix": "（具体修复建议）"
}
```

**问题级别**：
- **P0** — 必须修复：事实性错误、代码错误、路径错误、API 不存在、与 docs 直接矛盾
- **P1** — 建议修复：概念歧义、术语不一致、正则/格式错误
- **P2** — 可选修复：描述模糊、引用缺失

---

## 一、检查前准备

### 1.1 确认执行上下文

在开始检查前，必须确认以下要素：

| 上下文要素 | 说明 | 如何确认 |
|------------|------|----------|
| 工作目录 | 命令执行时的 pwd | 检查 SKILL.md 是否说明 |
| 环境变量 | 预设的变量如 `$SKILL_DIR` | 搜索 SKILL.md 中的变量定义 |
| 前置条件 | 如"已 cd 到某目录" | 检查工作流前置步骤 |

**判断方法**：

**步骤 1：检查 SKILL.md 是否说明了执行上下文**

```markdown
# 示例：明确说明工作目录
## 约定
执行以下命令前，请先 cd 到 <skill_dir>

# 示例：定义环境变量
- `$SKILL_DIR`：指向当前 skill 根目录
```

**步骤 2：检查同类 skill 的惯例**

```bash
# 检查其他 skill 是否有相同写法
grep -r "scripts/" .agents/skills/*/SKILL.md | wc -l
# 如果多个 skill 都这样写，说明是约定，不是错误
```

**步骤 3：验证资源是否存在（在正确的上下文中）**

```bash
# 错误做法：从项目根目录验证
ls scripts/xxx.py  # 不存在 → 误判为错误

# 正确做法：从 skill 目录验证
ls .agents/skills/<skill>/scripts/xxx.py  # 存在 → 不是错误
```

### 1.2 扫描 references 目录

获取所有文件列表，按类型分组：

| 文件类型 | 检查重点 |
|----------|----------|
| Markdown (.md) | 文本内容、API 引用、路径引用 |
| JSON/YAML | 结构有效性、字段正确性 |
| Python/Shell 脚本 | 语法、路径可移植性 |

### 1.3 提取 SKILL.md 知识内容

识别 `SKILL.md` 中需要检查的知识性内容：

| 内容类型 | 识别模式 |
|----------|----------|
| API 说明 | `pypto.xxx` API 名称及参数说明 |
| 路径说明 | 文件路径引用（如 `docs/zh/xxx.md`、`scripts/xxx.py`） |
| 术语定义 | 框架概念术语（tile、tensor、pass、codegen、ub、gm 等） |

**排除模式**（流程语义，不检查）：
- 工作流步骤描述（如"运行脚本""读取文件""检查结果"）
- 检查策略说明（如"Parse frontmatter, validate with regex"）
- Skill 内部定义（如评审规则、报告模板）

---

## 二、内容正确性检查

### 2.1 API 相关检查

**检查项**：API 名称、参数列表、参数类型、返回值

**验证方法**：

```bash
# 搜索 API 定义
grep -r "pypto\.<api>" docs/zh/api/

# 检查参数说明
grep -A 20 "def <api>" docs/zh/api/xxx.md
```

**判定标准**：
- **P0** — API 不存在：`grep` 在 docs 中找不到该 API
- **P0** — 与 docs 矛盾：参数说明与 docs 中的定义直接冲突
- **P1** — 概念歧义：示例中使用了真实 API 但声称其不存在（应使用虚构 API 名称）

**示例**：

```markdown
# P0 错误：API 不存在
- `pypto.compile` — 编译 API
# 证据：grep "pypto.compile" docs/zh/ 无结果
# 修复：PyPTO 编译入口是 @pypto.frontend.jit

# P1 歧义：使用真实 API 但说它不存在
**示例**：尝试调用 `pypto.reshape` 但该 API 不存在
# 修复：使用虚构 API 如 `pypto.nonexistent_op`
```

### 2.2 路径相关检查

**检查项**：文件路径存在性、相对路径正确性、头文件路径完整性

**验证方法**：

```bash
# 验证路径存在（在正确的上下文中）
ls <path> 2>/dev/null || echo "NOT_FOUND"

# 检查相对路径
cd <skill_dir> && ls <relative_path>
```

**判定标准**：
- **P0** — 路径不存在：在正确上下文中仍找不到文件
- **P0** — 相对路径错误：`../` 层级错误导致指向错误位置
- **P2** — 引用缺失：引用了不存在的资源文件（如 Excel）

**示例**：

```markdown
# P0 错误：相对路径错误
[common_errors.md](../common_errors.md)
# 问题：同级目录用 ../ 会指向上层
# 修复：[common_errors.md](./common_errors.md)
```

### 2.3 命令相关检查

**检查项**：命令格式、参数格式、选项值合法性

**验证方法**：

```bash
# 检查命令帮助
<command> --help

# 对比 docs 中的命令说明
grep -A 10 "<command>" docs/zh/cli/
```

**判定标准**：
- **P0** — 与 docs 矛盾：选项值与 docs 中定义的可选值列表冲突
- **P1** — 格式错误：`-f=value` vs `-f value` 不一致
- **P2** — 描述模糊：缺少关键参数说明

**示例**：

```markdown
# P0 矛盾：与 docs 定义冲突
# references 说
"禁止 --type=all"

# docs 说
--type | 可选：deps, cann, third_party, all

# 修复：移除"禁止"描述，或更新为正确的约束说明
```

### 2.4 代码相关检查

**检查项**：语法正确性、函数名正确性、常量名正确性

**验证方法**：
- 代码审查
- 如可执行，实际运行验证
- 对比官方示例或文档

**判定标准**：
- **P0** — 语法错误：括号不匹配、关键字拼写错误
- **P0** — 函数名错误：调用不存在的函数

**示例**：

```cpp
// P0 错误：函数名重复
EXPECT_EQ_EQ(pass.PostCheck(function), SUCCESS);

// 修复
EXPECT_EQ(pass.PostCheck(function), SUCCESS);
```

---

## 三、语义一致性检查

### 3.1 术语一致性

**检查项**：概念术语、组件名称、参数名称是否与 docs 一致

**验证方法**：

```bash
# 检查术语表
grep -i "<term>" docs/zh/tutorials/appendix/glossary.md

# 检查组件名称
grep -r "Tile Graph\|Tensor Graph" docs/zh/
```

**判定标准**：
- **P1** — 术语不一致：与 docs 术语表或常见表述不一致
- **PASS** — 合理变体：同一概念有多种等价表述

**示例**：

```markdown
# P1 不一致
references: "尾轴 32B 对齐"
docs: "外轴切分大小满足 32B 对齐" / "尾轴 32B 对齐"

# 建议：确认语境后统一表述，或两者都接受
```

### 3.2 约束一致性

**检查项**：硬件约束、dtype 支持、版本要求是否与 docs 一致

**验证方法**：

```bash
# 检查 dtype 支持
grep -A 20 "dtype" docs/zh/api/<api>.md

# 检查版本要求
grep -i "version\|cann\|pytorch" docs/zh/installation/
```

**判定标准**：
- **P0** — 与 docs 矛盾：约束条件与 docs 直接冲突
- **P2** — 描述模糊：简化写法可能遗漏重要信息

**示例**：

```markdown
# P2 模糊
- dtype: INT8-64

# 更清晰（或指向 docs）
- dtype: INT8/UINT8/INT16/UINT16/INT32/UINT32/INT64/UINT64
- dtype: 详见 docs/zh/api/others/pypto-from_torch.md
```

---

## 四、排除项（非问题）

以下情况**不需要标记为问题**：

| 类型 | 示例 | 原因 |
|------|------|------|
| 合理简化 | dtype: FP16/BF16/FP32/INT8-64/BOOL | 速查表 + 指向 docs |
| 更严格规范 | commit message 10-200 字符限制 | skill 可定义更严格要求 |
| 内部知识 | 常见错误排查经验、troubleshooting | docs 不一定包含 |
| 上下文相关路径 | scripts/xxx.py | 已确认执行上下文 |
| 无对应 docs | 评审规则、报告模板、skill 内部内容 | docs 无对应主题 |

---

## 五、联动修改检查

发现问题时，检查是否需要同步修改：

| 检查项 | 验证方法 |
|--------|----------|
| SKILL.md | `grep -n "<error_content>" SKILL.md` |
| 其他 references | `grep -rn "<error_content>" references/` |
| scripts/ | `grep -rn "<error_path>" scripts/` |

**联动修改标记**：若发现多处引用相同错误内容，在报告中标注"需要联动修改 N 处"。

---

## 六、问题分类与优先级

### P0 必须修复

| 类型 | 定义 | 识别方法 |
|------|------|----------|
| 代码错误 | 语法错误、拼写错误、函数名错误 | 代码审查、运行验证 |
| 路径错误 | 文件路径不存在、相对路径指向错误 | `ls <path>` 验证 |
| API 不存在 | 引用了 docs 中不存在的 API | `grep "pypto\.<api>" docs/zh/` |
| 与 docs 矛盾 | 参数说明、选项值等与 docs 直接矛盾 | 对比 docs 中相同主题 |

### P1 建议修复

| 类型 | 定义 | 识别方法 |
|------|------|----------|
| 概念歧义 | 术语使用可能造成误解 | 示例使用真实 API 但声称不存在 |
| 术语不一致 | 与 docs 术语表不一致 | `grep -i "<term>" docs/zh/tutorials/appendix/glossary.md` |
| 正则/格式错误 | 正则表达式、格式规范有语法问题 | 正则测试、格式验证 |

### P2 可选修复

| 类型 | 定义 | 识别方法 |
|------|------|----------|
| 描述模糊 | 简化写法可能遗漏重要信息 | 检查是否有指向 docs 的链接 |
| 引用缺失 | 引用了不存在的资源文件 | `ls <resource_file>` |

---

## 七、自检清单

在完成全部 references 检查后，执行以下校验：

1. **上下文确认**：所有路径错误都已确认不是执行上下文问题
2. **证据充分**：每个 P0/P1 问题都有 docs 中的具体证据或验证命令结果
3. **排除项应用**：已正确识别合理简化、更严格规范、内部知识等排除项
4. **联动检查**：已检查是否需要在其他文件中同步修改
5. **问题分级正确**：P0/P1/P2 分级符合定义标准
