# PyPTO Agent Skills

面向 PyPTO 框架开发与维护的专家技能集，覆盖**框架调试与错误定位、编译期 Pass 分析、环境配置、PR / Issue 流程**。同时支持 [opencode](https://opencode.ai) 与 [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview)：opencode 自动发现 `.agents/skills/` 与 `.opencode/agents/`，Claude Code 按下方 [Claude Code 配置](#claude-code-配置) 一次性迁移。

## Skills

### 调试与错误定位

| Skill | 用途 |
|:---|:---|
| [`pypto-aicore-error-locator`](skills/pypto-aicore-error-locator/SKILL.md) | aicore error 时定位 CCE 文件与源代码行 |
| [`pypto-host-stacktrace-analyzer`](skills/pypto-host-stacktrace-analyzer/SKILL.md) | host 端 Python/C++ 堆栈的地址—源码映射与符号解析 |
| [`pypto-memory-overlap-detector`](skills/pypto-memory-overlap-detector/SKILL.md) | MACHINE workspace 内存重叠与管理问题的检测与修复 |
| [`pypto-machine-workspace`](skills/pypto-machine-workspace/SKILL.md) | workspace 内存异常偏大的诊断；逐层拆解内存预算 |
| [`pypto-precision-overall`](skills/pypto-precision-overall/SKILL.md) | 算子精度问题场景路由式排查：用例剪枝 → tensor-graph → pass → 上板二分 |

### Pass 模块（编译期）

| Skill | 用途 |
|:---|:---|
| [`pypto-pass-error-locator`](skills/pypto-pass-error-locator/SKILL.md) | Pass 模块错误诊断：定位、根因分析、修复建议 |
| [`pypto-pass-module-analyzer`](skills/pypto-pass-module-analyzer/SKILL.md) | Pass 模块代码分析；产出结构化 Pass 模块文档 |
| [`pypto-pass-workflow-analyzer`](skills/pypto-pass-workflow-analyzer/SKILL.md) | Pass 业务流分析：职责、执行顺序、数据依赖 |
| [`pypto-pass-perf-optimizer`](skills/pypto-pass-perf-optimizer/SKILL.md) | Pass 编译期性能优化 |
| [`pypto-pass-ut-generate`](skills/pypto-pass-ut-generate/SKILL.md) | 根据业务描述生成 Pass UT 用例 |

### PR / Issue / 代码质量

| Skill | 用途 |
|:---|:---|
| [`pypto-pr-creator`](skills/pypto-pr-creator/SKILL.md) | 创建/更新 `cann/pypto` PR：fork 校验、Git 认证、upstream 同步、commit 检查、CLA |
| [`pypto-pr-fixer`](skills/pypto-pr-fixer/SKILL.md) | 修复 PyPTO PR 的 CodeCheck CI 失败与 review 评论 |
| [`pypto-issue-creator`](skills/pypto-issue-creator/SKILL.md) | 创建结构化 GitCode Issue：Bug、Feature、Doc、Question 等 |
| [`pypto-fracture-point-detector`](skills/pypto-fracture-point-detector/SKILL.md) | 识别框架/文档断裂点，产出可转化为 Issue 的报告 |
| [`pypto-skill-reviewer`](skills/pypto-skill-reviewer/SKILL.md) | 审计 skill 目录的质量与最佳实践合规性，并打分 |
| [`pypto-skill-validation-prompt`](skills/pypto-skill-validation-prompt/SKILL.md) | 为任意 skill 生成校验提示词，验证产物是否匹配自身声明 |

### 环境与安装

| Skill | 用途 |
|:---|:---|
| [`pypto-environment-setup`](skills/pypto-environment-setup/SKILL.md) | PyPTO 环境安装与修复：CANN、torch_npu、工具链、第三方依赖 |
| [`gitcode-mcp-install`](skills/gitcode-mcp-install/SKILL.md) | 安装与配置 GitCode MCP Server，使 AI 客户端可与 GitCode 平台交互 |

## Agent

**`pypto-code-merge-agent`** — 代码合并助手：检测已暂存的变更、推测修改意图、生成规范的 commit / Issue / PR，并按序编排 `pypto-issue-creator` → `pypto-pr-creator` 完成提交。触发词：`go`、`提交PR`、`创建PR`、`merge`。

## Claude Code 配置

Claude Code 使用 `.claude/` 目录，需一次性从 opencode 结构迁移：

```bash
mkdir -p .claude/skills .claude/agents
cp AGENTS.md CLAUDE.md
cp -r .agents/skills/* .claude/skills/
cp -r .opencode/agents/* .claude/agents/
```

> [!NOTE]
> `mode: subagent` 是 opencode 专有字段，Claude Code 会忽略；如遇告警可去除：
> `for f in .opencode/agents/*.md; do sed '/^mode:/d' "$f" > ".claude/agents/$(basename "$f")"; done`

## 目录结构

```
AGENTS.md            项目入口（通用原则 + 入口路径速查）
.agents/
└─ skills/           专家技能
.opencode/
└─ agents/           pypto-code-merge-agent
```
