# GitCode 行级评论 API（diff_comment）

通过 GitCode API 将检视意见提交到 PR 的**具体代码行**，显示在 diff 视图对应行旁（而非 PR 级普通评论）。

## 认证

从 git remote 提取 token（无需额外配置）：
```bash
TOKEN=$(git remote get-url origin | sed -nE 's|.*oauth2:([^@]+)@.*|\1|p')
```

三种认证方式均可：
- `PRIVATE-TOKEN: <token>`（header）
- `Authorization: Bearer <token>`（header）
- `?access_token=<token>`（query param）

验证：
```bash
curl -s -o /dev/null -w "%{http_code}" "https://gitcode.com/api/v5/user?access_token=${TOKEN}"
# 200 = 可用
```

---

## 核心 API

### POST 行级评论

```
POST /api/v5/repos/:owner/:repo/pulls/:number/comments
```

**正确请求格式**（关键：`position` 字段，不是 `line`/`side`/`diff_side`）：

```json
{
  "body": "检视意见内容",
  "path": "framework/src/machine/runtime/bundle/kernel_bundle_crc32.h",
  "position": 31,
  "need_to_resolve": true
}
```

**成功响应特征**（`comment_type` 必须是 `diff_comment`）：

```json
{
  "id": 182608753,
  "comment_type": "diff_comment",
  "diff_position": {
    "start_new_line": 31,
    "end_new_line": 31,
    "position_type": "text"
  },
  "body": "...",
  "resolved": false,
  "created_at": "...",
  "user": {...}
}
```

> ⚠️ POST 响应体只返回 `{id, body, note_id}` 三个字段，**不含** `comment_type`/`diff_position`。
> 必须通过 GET 列表验证是否真正成为 diff_comment（见下方"验证"）。

### curl 示例

```bash
curl -s -H "PRIVATE-TOKEN: ${TOKEN}" -H "Content-Type: application/json" \
  -X POST "https://gitcode.com/api/v5/repos/cann/pypto/pulls/4974/comments" \
  -d "$(jq -n --arg b "$BODY" --arg p "$PATH" --argjson pos $LINE \
    '{body:$b, path:$p, position:$pos, need_to_resolve:true}')"
```

### 验证是否成为 diff_comment

```
GET /api/v5/repos/:owner/:repo/pulls/:number/comments?per_page=50
```

```bash
curl -s -H "PRIVATE-TOKEN: ${TOKEN}" \
  "https://gitcode.com/api/v5/repos/cann/pypto/pulls/4974/comments?per_page=50" | \
  python3 -c "
import sys, json
for c in json.load(sys.stdin):
    if c.get('comment_type') == 'diff_comment':
        dp = c.get('diff_position', {})
        print(f\"id={c['id']} line={dp.get('start_new_line')} body={c['body'][:40]}...\")
"
```

**判定标准**：
- `comment_type == "diff_comment"` 且 `diff_position` 非空 → ✅ 行级评论
- `comment_type == "pr_comment"` 或 `diff_position` 为 None → ❌ 降级为 PR 级普通评论

---

## 字段对照表（踩坑记录）

| 字段 | 是否生效 | 结果 | 说明 |
|------|---------|------|------|
| **`position`** | ✅ **正确** | `diff_comment` + `diff_position` | **唯一能产生行级评论的字段** |
| `line` | ❌ 无效 | `pr_comment`（降级） | POST 返回 201 但不关联到行 |
| `side` (`RIGHT`/`LEFT`) | ❌ 无效 | `pr_comment`（降级） | Gitea/GitHub 字段，GitCode 不支持 |
| `diff_side` (`new`/`old`) | ❌ 无效 | `pr_comment`（降级） | GitHub 字段，GitCode 不支持 |
| `commit_id` | ❌ 无效 | 不影响 comment_type | 传了也不报错但无作用 |
| `need_to_resolve` | ✅ 可选 | 设置 resolved 状态 | true = 标记需解决 |

**结论**：只用 `body` + `path` + `position` + `need_to_resolve` 四个字段。

---

## 其他相关 API

### GET PR 信息

```
GET /api/v5/repos/:owner/:repo/pulls/:number
```
获取 `head.sha`、`base.sha`、关联 issue number。

### GET PR 关联 issue

```
GET /api/v5/repos/:owner/:repo/pulls/:number/issues
```
返回 `[{number, id, ...}]`，用于 issue 级评论提交。

### POST issue 级评论（PR 级普通评论，非行级）

```
POST /api/v5/repos/:owner/:repo/issues/:issue_number/comments
```
```json
{"body": "整体检视意见汇总..."}
```
用于提交整体 review 总结（非行级）。

### DELETE 评论

```
DELETE /api/v5/repos/:owner/:repo/pulls/comments/:id
```
> ⚠️ GitCode API 不支持 DELETE（返回 405 Method Not Allowed）。测试评论无法通过 API 删除，需在网页手动删除。

### PATCH 更新评论

```
PATCH /api/v5/repos/:owner/:repo/pulls/comments/:id
```
> ⚠️ 同样不支持（405）。提交前确保内容正确。

---

## 常见错误

| 现象 | 原因 | 修复 |
|------|------|------|
| POST 返回 201 但评论不在 diff 行上 | 用了 `line`/`side` 而非 `position` | 改用 `position` 字段 |
| `comment_type` 为 `pr_comment` | 同上 | 同上 |
| 404 Not Found | PR number 错误或无权限 | 确认 owner=cann repo=pypto |
| 401 Unauthorized | token 无效或过期 | 重新从 git remote 提取 |
| 行号偏移 | 用了 base 行号或 diff 相对行号 | 用 `git show $PR_HEAD:<file> \| grep -n` 取 head 绝对行号 |

---

## 行号获取方法

行号必须是 **PR head 文件中的绝对行号**（不是 base 行号，不是 diff hunk 中的相对行号）：

```bash
PR_HEAD=$(git rev-parse FETCH_HEAD)

# 方法 1：grep 关键模式
git show "${PR_HEAD}:framework/src/machine/runtime/bundle/kernel_bundle_crc32.h" | grep -n "0xEDB88420"

# 方法 2：带上下文查看
git show "${PR_HEAD}:<file>" | sed -n '28,35p' | cat -n
```
