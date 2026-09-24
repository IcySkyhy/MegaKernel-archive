---
name: pypto-pr-creator
description: 将本地修改创建或更新到 cann/pypto 仓库的 PR。覆盖 fork 验证、Git 认证、upstream 同步、commit 规范检查、PR 创建/更新和 CLA 检查。当用户提到创建 PR、提交代码、代码合并、推送到 cann/pypto，或已完成开发需要提交变更时使用。触发词：创建PR、修改更新、提交PR、更新PR、修改PR、PR规范、commit message、pypto贡献、提交代码。
---

# PyPTO PR Creator

从用户 fork 仓库向 `cann/pypto` 创建 PR 的完整流程。

```
<username>/pypto  ──PR──▶  cann/pypto
    (用户 fork)              (上游主仓库)
```

## 环境依赖

### GitCode MCP

本 skill 的所有远程操作（fork 验证、PR 查询/创建/更新）均通过 GitCode MCP 完成。

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

## 强制约束

- PR 目标必须是 `cann/pypto`，不是用户 fork
- 远程操作通过 GitCode MCP 完成，禁止直接使用 `GITCODE_TOKEN` 进行 API 调用
- `origin` 指向用户 fork，禁止指向 `cann/pypto`
- 禁止打印 `GITCODE_TOKEN`，包括屏幕、日志、调试信息
- 创建分支、commit、push、创建/更新 PR 前必须获得用户明确确认
- Commit message 使用英文，格式 `tag(scope): Summary`，整个 message 不超过 10 行，不以句号结尾，首字母大写，长度 10-200 字符

## 参考文件

| 文件 | 用途 | 加载时机 |
|------|------|----------|
| [references/pr-spec.md](references/pr-spec.md) | PR 标题、Body、Commit 的格式规范 | 编写 commit message 和 PR 内容时读取 |
| [references/checklist.md](references/checklist.md) | 提交前逐项检查清单 | 阶段 4（预检）和阶段 6（创建 PR）前读取 |
| [references/git-auth.md](references/git-auth.md) | Git 认证方式配置与排查 | 阶段 2 检测到认证缺失时读取 |
| [references/troubleshooting.md](references/troubleshooting.md) | push/PR 创建失败诊断与修复 | 遇到具体错误时按需读取 |

---

## 工作流（6 个阶段）

### 阶段 1：仓库发现与验证

1. 检查当前工作目录是否为 pypto 仓库（`git rev-parse --is-inside-work-tree`）
2. 验证 origin 指向用户 fork（包含 `pypto` 且不包含 `cann/pypto`）
3. 若当前目录不符合，在工作区搜索 `.git` 目录并检查 remote
4. 检测浅克隆（`git rev-parse --is-shallow-repository`），标记以便后续处理
5. 通过 `gitcode_get_repository(owner="<username>", repo="pypto")` 验证 fork 链（`parent.full_name == "cann/pypto"`）

**失败路径**：origin 指向 `cann/pypto` 时，执行 `git remote set-url origin https://gitcode.com/<username>/pypto.git` 修复。

### 阶段 2：Git 认证检查

push 需要 Git 认证。快速检测：

```bash
echo "GITCODE_TOKEN: $([ -n "$GITCODE_TOKEN" ] && echo '已设置' || echo '未设置')"
echo "credential.helper: $(git config --global credential.helper 2>/dev/null || echo '未配置')"
```

- 已配置认证 → 继续
- 未配置 → 读取 [references/git-auth.md](references/git-auth.md)，展示认证方式选项，等待用户选择并配置

> 认证配置成功前，禁止执行 push 操作。

### 阶段 3：用户确认

向用户展示执行计划表，等待明确确认：

| 确认项 | 内容 |
|--------|------|
| 本地仓库路径 | `$PYPTO_REPO` |
| Fork 仓库 | `<username>/pypto` |
| 分支名 | `<branch_name>` |
| Commit 信息 | `tag(scope): Summary` |
| Push 目标 | origin → `<branch_name>` |
| PR 目标 | `cann/pypto` → `master` |
| PR 标题与 Body | 预览内容 |

> 在获得用户明确确认之前，禁止执行任何 git 操作。

### 阶段 4：预检与提交

读取 [references/checklist.md](references/checklist.md) 执行预检：

1. **浅克隆修复**：`git fetch --unshallow origin`
2. **Upstream 同步**：`git fetch upstream master`，检查分支是否落后，落后则 `git rebase upstream/master && git push -f origin <branch>`
3. **Commit message 格式验证**：`git log -1 --format="%s" | grep -E '^(feat|fix|docs|style|refactor|perf|test)(.*): [A-Z].{10,200}'`

执行 git 操作：

```bash
git -C "$PYPTO_REPO" checkout -b <branch_name>
git -C "$PYPTO_REPO" add <files>
git -C "$PYPTO_REPO" commit -m "tag(scope): Summary"
git -C "$PYPTO_REPO" push origin <branch_name>
```

> push 失败时读取 [references/troubleshooting.md](references/troubleshooting.md) 按错误类型排查。

### 阶段 5：创建或更新 PR

读取 [references/pr-spec.md](references/pr-spec.md) 获取格式规范。

**判断创建还是更新**：

```python
prs = gitcode_list_pull_requests(owner="cann", repo="pypto")
# 筛选 state == "opened" 且 source_branch 匹配当前分支
# 存在 → 询问用户：更新现有 PR 还是创建新 PR
# 不存在 → 创建新 PR
```

**创建 PR**：

```python
gitcode_create_pull_request(
    owner="cann",
    repo="pypto",
    title="tag(scope): Summary",
    head="<username>:<branch_name>",
    base="master",
    body="..."
)
```

**更新 PR**：

```python
gitcode_update_pull_request(
    owner="cann",
    repo="pypto",
    pull_number=<pr_number>,
    title="新标题",
    body="新描述"
)
```

> MCP 返回 400 时，读取 [references/troubleshooting.md](references/troubleshooting.md) 使用 curl 获取详细错误。

### 阶段 6：PR 后检查

**PR 链接验证**：确认链接格式为 `https://gitcode.com/cann/pypto/merge_requests/<pr_id>`（不是 `<username>/pypto/...`）。

**CLA 检查**：

```python
pr = gitcode_get_pull_request(owner="cann", repo="pypto", pull_number=<pr_number>)
labels = pr.get("labels", [])
# 检查是否含 cla/no → CLA 未通过
```

CLA 失败时，读取 [references/troubleshooting.md](references/troubleshooting.md) 执行 CLA 修复流程。

**输出结构化报告**：PR 链接、操作类型、标题、源分支→目标分支、commit hash、后续操作提示。

---

## 完成标准

- PR 链接指向 `cann/pypto`（非用户 fork）
- Commit message 格式符合 `tag(scope): Summary` 正则
- CLA 检查通过或已向用户提供修复指引
- 向用户输出了包含 PR 链接的结构化报告

## 相关文件

| 文件 | 路径 |
|------|------|
| 贡献指南 | `$PYPTO_REPO/CONTRIBUTION.md` |
| PR 模板 (EN) | `$PYPTO_REPO/.gitcode/PULL_REQUEST_TEMPLATE_en.md` |
| PR 模板 (CN) | `$PYPTO_REPO/.gitcode/PULL_REQUEST_TEMPLATE.md` |
| 详细规范 | `$PYPTO_REPO/docs/zh/contribute/pull-request.md` |
| 代码检查规则 | `$PYPTO_REPO/docs/zh/contribute/code-check-rule.yaml` |
