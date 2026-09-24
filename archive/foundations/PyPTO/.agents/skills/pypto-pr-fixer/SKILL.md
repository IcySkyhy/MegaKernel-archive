---
name: pypto-pr-fixer
description: 修复 PyPTO PR 的 CodeCheck CI 失败和 review 评论。自动获取 CodeCheck 违规详情、匹配规则、应用修复。触发词：修复codecheck、codecheck问题、codecheck报错、codecheck失败、codecheck不通过、CI失败、CI报错、PR评论修复、review意见修复、修复PR、PR review fixer。
---

# PyPTO PR Fixer

PyPTO 仓库 PR 自动修复工具，提供两大能力：

## 环境依赖

| 依赖 | 版本 | 安装命令 |
|------|------|----------|
| Python | 3.10+ | 系统自带 |
| playwright | 1.58+ | `pip install playwright` |
| Chromium Headless Shell | v1208 | `playwright install chromium-headless-shell` |
| GitCode MCP | — | 调用 `gitcode-mcp-install` skill 配置 |

**注意**：ARM64 环境只能用 `chromium-headless-shell`，不能用完整 Chrome。

### GitCode MCP 配置检查

本 skill 的 PR 评论获取和 CI 状态检查依赖 GitCode MCP 工具。

检查方式：

```bash
CONFIG_FILE="$HOME/.config/opencode/opencode.json"

if [ -f "$CONFIG_FILE" ]; then
    TOKEN_VALUE=$(cat "$CONFIG_FILE" | grep -oP '"GITCODE_TOKEN"\s*:\s*"\K[^"]+' 2>/dev/null || echo "")
    if [ -n "$TOKEN_VALUE" ] && [ "$TOKEN_VALUE" != "<YOUR_GITCODE_TOKEN>" ]; then
        echo "GITCODE_TOKEN_STATUS=CONFIGURED"
    else
        echo "GITCODE_TOKEN_STATUS=NOT_CONFIGURED"
    fi
else
    echo "GITCODE_TOKEN_STATUS=CONFIG_FILE_NOT_FOUND"
fi
```

- **CONFIGURED** → 正常执行
- **NOT_CONFIGURED / CONFIG_FILE_NOT_FOUND** → 调用 `gitcode-mcp-install` skill 引导用户完成配置，等待执行完成后提示用户需要：
  1. 在 `~/.config/opencode/opencode.json` 中将 `<YOUR_GITCODE_TOKEN>` 替换为真实 token
  2. 重启 OpenCode 使配置生效

---

## 输入协议

### 模式 1: PR URL

```
https://gitcode.com/<owner>/<repo>/pull/<number>
https://gitcode.com/<owner>/<repo>/merge_requests/<number>
```

### 模式 2: 三元组

```
owner: cann
repo: pypto
pull_number: 1276
```

### 本地仓库位置

- 默认：当前工作目录（须为 git 仓库）
- pypto 仓库推荐路径：`/workspace/opencode/pypto_repo`

## 核心流程

```
1. 解析 PR 标识 → owner/repo/number
2. 检查 CI 标签状态
   ├── ci-pipeline-running → 等待 3 分钟后重试（循环）
   ├── ci-pipeline-failed → 先判定失败根因（codecheck / 非 codecheck）
   └── ci-pipeline-passed → 继续处理
3. 获取 PR 元数据和全部评论（使用 GitCode MCP 工具）
4. 过滤机器人评论，分离人工评论与 CI 报告
5. 对人工评论：通用理解 → 定位文件 → 生成修复方案
6. 对 codecheck 失败：
   ├── 6a. 判定失败根因并提取报告 URL — 使用 scripts/extract_latest_codecheck_url.py
   ├── 6b. 获取违规详情 — 使用 scripts/fetch_codecheck_violations.py
   └── 6c. 匹配规则修复 — 使用 scripts/query_codecheck_rule.py
7. 用户确认修复方案
8. 应用修复
9. 本地预检 — 使用 scripts/local_codecheck.py 扫描
   └── 发现问题 → 返回步骤 8 继续修复
   └── 无问题 → 继续

> **⚠️ 开始前**：
> 1. 切换到 PR 对应的本地分支：`git checkout <branch_name>`
> 2. 确认已配置 upstream remote：`git remote -v | grep upstream || git remote add upstream https://gitcode.com/cann/pypto.git`

10. 同步 upstream（检查 + rebase）
11. 委托 pypto-pr-creator 完成 commit + push
```

## 评论获取与分类

### 获取评论（使用 GitCode MCP 工具）

使用 GitCode MCP 工具获取 PR 评论，禁止手动构造 HTTP 请求：

```python
# 1. 获取 PR 元数据（包含标签信息）
pr_info = gitcode_get_pull_request(owner, repo, pull_number)

# 2. 获取全部评论（包含 pr_comment 和 diff_comment）
comments = gitcode_list_pull_request_comments(
    owner=owner,
    repo=repo,
    pull_number=pull_number,
    limit=0,  # 必须拉全量评论，避免仅返回默认20条导致遗漏最新CI结果
    comment_type="all",
)
```

## CI 状态检查

### 检查 PR 标签

获取 PR 元数据后，首先检查 CI 标签状态：

```python
import time
import logging

def wait_for_ci_complete(owner: str, repo: str, pull_number: int) -> dict:
    """循环等待直到 ci-pipeline-running 标签消失。"""
    while True:
        pr_info = gitcode_get_pull_request(owner, repo, pull_number)
        labels = [label.get("name", "") for label in pr_info.get("labels", [])]

        if "ci-pipeline-running" not in labels:
            return pr_info

        logging.info("检测到 ci-pipeline-running 标签，CI 仍在运行")
        logging.info("等待 3 分钟后重新检查...")
        time.sleep(180)
```

### 标签处理逻辑

| 标签状态 | 处理方式 |
|---------|----------|
| `ci-pipeline-running` | 循环等待（每 3 分钟检查一次，直到标签消失） |
| `ci-pipeline-failed` | 先判定失败根因；仅在 codecheck 失败时走 codecheck 修复流程 |
| `ci-pipeline-passed` | 无需处理 CI，仅处理人工 review 评论 |
| 标签消失后 | 基于最新状态继续后续流程 |

### 机器人评论过滤

```python
def is_robot_comment(comment):
    login = comment.get("user", {}).get("login", "").lower()
    if login == "cann-robot":
        return True
    return any(kw in login for kw in ["bot", "robot", "ci", "automation"])
```

### 评论分流

将评论分为两类：

1. **cann-robot CI 报告** — 包含 HTML 表格（`codecheck ❌ FAILED`），走 CodeCheck 修复流程
2. **人工 review 评论** — 其余所有评论，走通用修复流程

---

## 能力一：通用 Review 评论修复

### 设计原则

**不预设固定分类**。LLM 直接理解评论语义，动态判断修复方案。

### 通用修复处理流程

1. **理解意图** — 阅读评论全文，理解 reviewer 要求什么
2. **判断可行性** — 该修改是否可以自动执行？
   - 可自动修复：明确的代码修改指令（改名、移动字段、修复格式、添加缺失内容等）
   - 需人工判断：涉及设计决策、业务逻辑、架构变更等
3. **定位文件** — 通过以下方式定位受影响文件：
   - `diff_comment` 的 `diff_position`（行号信息）
   - 评论中提及的文件路径
   - `grep` 搜索评论中提到的关键内容
4. **生成修复** — 使用 `edit`/`write` 工具直接修复
5. **标记待办** — 无法自动修复的评论输出为结构化 TODO

### 输出格式

```yaml
# 可自动修复
- comment_id: 164054117
  intent: "将 source_url 从顶层移至 metadata 下"
  auto_fixable: true
  files:
    - path/to/SKILL.md
  action: "移动 YAML 字段位置"

# 需人工处理
- comment_id: 164054200
  intent: "重新设计 API 接口"
  auto_fixable: false
  reason: "涉及架构决策"
  suggested_action: "与 reviewer 讨论方案后再修改"
```

### diff_comment 注意事项

GitCode MCP 的 `diff_comment` 包含 `diff_position`（`start_new_line`/`end_new_line`），但**不包含文件 `path`**。定位文件时需结合 grep 搜索评论提及的代码内容。

详见 [references/review-guide.md](references/review-guide.md)。

## 能力二：CodeCheck CI 修复

### 触发条件

cann-robot 评论中包含 codecheck 失败的 HTML 表格：

```html
<tr>
  <td><strong>codecheck</strong></td>
  <td>❌ FAILED</td>
  <td><a href="https://www.openlibing.com/apps/entryCheckDashCode/{MR_ID}/{hash}?projectId=300033&codeHostingPlatformFlag=gitcode">>>>>>></a></td>
</tr>
```

### CodeCheck 修复处理流程

使用 `scripts/extract_latest_codecheck_url.py` 先判定最新 CI 是否由 codecheck 导致失败，再决定输出：

```bash
# 推荐：先做最新 CI 判定
python scripts/extract_latest_codecheck_url.py \
  --input comments.json \
  --gate-on-latest-ci \
  --format json

# codecheck 失败时输出证据链
python scripts/extract_latest_codecheck_url.py \
  --input comments.json \
  --gate-on-latest-ci \
  --evidence \
  --format json
```

**判定结果分支**：
- `kind=codecheck_failed`：输出 `codecheck_url`；`--evidence` 时额外输出证据链
- `kind=non_codecheck_failed`：输出 CI 状态和失败任务，不输出 codecheck URL
- `kind=undecidable`：输出无法判定原因（例如未找到可解析 CI 表格）

**非 codecheck 失败示例**（`kind=non_codecheck_failed`）：
```json
{
  "kind": "non_codecheck_failed",
  "codecheck_status": "✅ SUCCESS",
  "failed_tasks": [
    {"task": "UT_Test_Cpp_make_gnu_part_2", "status": "❌ FAILED"}
  ]
}
```

**证据链输出**（`--evidence` 参数）：
```json
{
  "total_found": 3,
  "latest": {
    "comment_id": 164054130,
    "created_at": "2026-03-07T10:30:00Z",
    "url": "https://www.openlibing.com/apps/entryCheckDashCode/..."
  },
  "evidence_chain": [
    {"index": 1, "comment_id": 164054130, "created_at": "2026-03-07T10:30:00Z", "url": "...", "is_latest": true},
    {"index": 2, "comment_id": 164054100, "created_at": "2026-03-07T09:15:00Z", "url": "...", "is_latest": false},
    {"index": 3, "comment_id": 164054080, "created_at": "2026-03-07T08:00:00Z", "url": "...", "is_latest": false}
  ]
}
```

**证据链说明**：
- `total_found`: 共找到多少个 codecheck URL
- `latest`: 最新的 URL 及其元信息
- `evidence_chain`: 完整证据链，按时间倒序排列，便于审计和验证

**降级条件**（仅以下情况可手动解析）：
1. 脚本文件不存在
2. 脚本执行报错且无法修复
3. 用户明确指定使用其他方式

#### 步骤 2：获取违规详情

使用 `scripts/fetch_codecheck_violations.py` 从 codecheck URL 获取违规列表：

```bash
python scripts/fetch_codecheck_violations.py "$CODECHECK_URL" --output json
```

脚本默认优先走 **task API 分页抓取**（按 `pageNum/pageSize` 拉全量），仅在 API 不可用时降级为 DOM 解析。
当页面概览为 156 时，输出 `total` 也应为 156（而非单页 20）。

**降级条件**（仅以下情况可使用其他方式）：
1. 脚本文件不存在
2. 脚本执行报错且无法修复
3. 用户明确指定使用其他方式

降级时需记录原因并告知用户。

> **注意**：openlibing.com 是 SPA，受 WAF 保护，Playwright MCP 不支持 ARM64，脚本内部使用 Playwright Python 实现。

#### 步骤 3：查询规则修复方案

使用 `scripts/query_codecheck_rule.py` 查询违规规则的修复方案：

```bash
python scripts/query_codecheck_rule.py \
  --from-violations-json violations.json \
  --language python \
  --format markdown
```

**降级条件**（仅以下情况可使用其他方式）：
1. 脚本文件不存在
2. 脚本执行报错且无法修复
3. 规则 ID 在脚本数据库中不存在
4. 用户明确指定使用其他方式

降级方案：查阅 `references/codecheck-rules.md` 或官方文档。

#### 步骤 4：分类处理

- 可自动修复（格式类 G.FMT、命名类 G.NAM、日志类 G.LOG 等）→ 直接修复
- 需人工判断（安全类 G.EDV、业务逻辑类 G.CTL 等）→ 生成修复建议

#### 步骤 5：应用修复

根据分类结果应用修复。

> **本地预检** 在核心流程的 **步骤 9** 执行（commit 前），此处不做。



## 本地预检（必须）

**执行时机**：修复完成后、commit 之前（核心流程步骤 9）

```bash
python scripts/local_codecheck.py <repo_path> --output json
```

**处理逻辑**：
- 发现问题 → 返回修复阶段继续处理
- 无问题 → 继续执行 commit

**不可跳过**：此步骤为提交前的最后保障，确保本地代码符合 CodeCheck 规则。

---


### CodeCheck 规则参考

完整规则映射见 [references/codecheck-rules.md](references/codecheck-rules.md)，包含 111 条 Python 规则的修复方案分类。

## PR 策略选择

获取 PR 信息后，确定修复提交策略：

| 条件 | 策略 | 操作 |
|------|------|------|
| 文件存在于 target_branch | `new_pr` | 从 target_branch 创建新分支，独立 PR |
| 文件仅存在于 source_branch | `append` | 追加到现有 PR 的 source_branch |
| 混合场景 | `append` | 优先追加，避免拆分修复 |

**用户确认**：向用户展示策略选择结果，等待确认后继续。

## 提交与 PR

### Commit Message 格式

```
fix(skills): <summary>

- <change 1>
- <change 2>
```

> **⚠️ Commit Message 格式（Pre-receive Hook 强制验证）**：`^(feat|fix|docs|style|refactor|perf|test)(.*): [A-Z].{10,200}` — Tag 必须是这 7 种之一，冒号后必须有空格，Summary 首字母必须大写且长度 10-200 字符。验证：`git log -1 --format="%s" | grep -E '^(feat|fix|docs|style|refactor|perf|test)(.*): [A-Z].{10,200}'`

### 同步 Upstream（关键步骤）

**在委托 pypto-pr-creator 之前，必须确保分支与 upstream 同步**，否则 push 会因 "pre receive hook check failed" 失败。

```bash
# 步骤 1：获取 upstream 最新状态
git fetch upstream master

# 步骤 2：检查分支是否落后
git log --oneline HEAD..upstream/master | wc -l
# 若输出 > 0，说明分支落后，需要 rebase

# 步骤 3：rebase（如落后）
git rebase upstream/master
git push -f origin <branch_name>
```


### 委托 pypto-pr-creator

PR 创建逻辑**完全委托给 pypto-pr-creator 技能**，传递：
- 本地仓库路径
- commit 信息
- PR 标题和描述

> **委托时已同步 upstream**，pypto-pr-creator 可直接执行 commit + push。
### PR 创建后检查

PR 创建成功后检查 CLA 和 LGTM 状态：
- CLA 未签署 → `⚠️ WARNING`（不阻塞提交，仅影响合并）
- LGTM 不足 → `ℹ️ INFO`（合并通常需要 ≥ 2 个 LGTM）

## PR 创建失败诊断

当遇到 `pre receive hook check failed` 时，执行诊断：

| 检查项 | 诊断命令 | 修复建议 |
|--------|----------|----------|
| 分支同步状态 | `git log HEAD..origin/<target> --oneline` | `git pull --rebase` |
| 提交者身份 | `git log -1 --format="%ae"` | 配置 `git user.email` |



## 限制说明

1. **diff_comment 不含文件路径** — 通过 grep 定位，可能存在误匹配
2. **openlibing.com 为 SPA** — 必须使用 Playwright 渲染，无法 HTTP 直接获取
3. **隐私保护** — 禁止打印 `GITCODE_TOKEN`（屏幕、日志、调试信息均禁止）

## 参考文档

- [通用修复指南](references/review-guide.md) — 评论理解与修复策略
- [CodeCheck 规则参考](references/codecheck-rules.md) — CodeArts-Check 规则映射与修复方案
- [错误处理参考](references/error-handling.md) — MCP 错误、平台错误、pre-receive hook 诊断
