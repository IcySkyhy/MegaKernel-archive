# 常见问题诊断与修复

> 遇到 push 失败或 PR 创建错误时按需加载。

## 目录

- [Push 失败](#push-失败)
  - [浅克隆报错](#浅克隆报错-shallow-update-not-allowed)
  - [认证错误 401/403](#认证错误-401403)
  - [分支落后 upstream](#分支落后-upstreampre-receive-hook-check-failed)
- [PR 创建失败](#pr-创建失败)
  - [MCP 返回 400](#mcp-返回-400)
  - [head 参数格式错误](#head-参数格式错误)
  - [pre-receive hook check failed](#pre-receive-hook-check-failed)
- [CLA 检查失败](#cla-检查失败)
- [Origin 配置错误](#origin-配置错误)

---

## Push 失败

### 浅克隆报错 "shallow update not allowed"

```bash
git fetch --unshallow origin
```

### 认证错误 401/403

**原因**：GitCode 不支持 Bearer token 认证，只支持 HTTP Basic Auth。

```bash
# 1. 删除错误的 Bearer token 配置
git config --local --unset http.extraheader

# 2. 配置正确的 credential helper
git config --local credential.helper store

# 3. 存储凭据（HTTP Basic Auth）
git credential-store store << 'EOF'
protocol=https
host=gitcode.com
username=<your_username>
password=<your_token>
EOF
```

**调试认证问题**：

```bash
GIT_CURL_VERBOSE=1 git push origin <branch_name> 2>&1 | grep -i authorization
# 正确: Authorization: Basic <base64>
# 错误: Authorization: Bearer <token>
```

### 分支落后 upstream（pre-receive hook check failed）

```bash
git fetch upstream master
git rebase upstream/master
git push -f origin <branch>
```

## PR 创建失败

### MCP 返回 400

MCP 工具可能吞掉详细错误信息，使用 curl 获取详情：

```bash
response=$(curl -s -w "\n%{http_code}" -X POST "https://api.gitcode.com/api/v5/repos/cann/pypto/pulls" \
  -H "PRIVATE-TOKEN: ${GITCODE_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"title": "tag(scope): Summary", "head": "<username>:<branch>", "base": "master", "body": "PR 描述"}')

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

if [ "$http_code" = "201" ]; then
    echo "PR 创建成功"
    echo "$body" | grep -o '"html_url":"[^"]*"' | cut -d'"' -f4
else
    echo "PR 创建失败 (HTTP $http_code): $body"
fi
```

### head 参数格式错误

```python
head="feat/add-pr-guide"              # 错误：缺少 fork owner
head="<username>/feat/add-pr-guide"   # 错误：用了 / 而非 :
head="<username>:feat/add-pr-guide"   # 正确
```

### pre-receive hook check failed

两个常见原因：
1. commit message 不符合 `tag(scope): Summary` 格式
2. 分支落后于 upstream（参见上方"分支落后 upstream"）

## CLA 检查失败

PR 标签含 `cann-cla/no` 时，按以下步骤排查：

**1. 检查 Git 配置与 GitCode 账户邮箱一致性**

```bash
git log -1 --format='Author: %an <%ae>'
git config --global user.email
```

CLA 检查基于 commit 作者邮箱，必须与 GitCode 账户主邮箱一致。

**2. 修复方式**

```bash
# 方式 1: 修改全局配置
git config --global user.name "your_username"
git config --global user.email "your_email@example.com"

# 方式 2: 修改最近一次 commit 的作者（已 push 需 force push）
git commit --amend --author="Your Name <your_email@example.com>"
git push -f origin <branch_name>
```

或在 GitCode → Settings → Emails 中添加 commit 使用的邮箱并验证。

**3. 重新触发 CLA 检查**

```bash
git commit --allow-empty -m "docs: trigger CLA check"
git push origin <branch_name>
```

## Origin 配置错误

| Origin 配置 | 判定 | 处理 |
|-------------|------|------|
| `<username>/pypto`（用户 fork） | 正确 | 直接使用 |
| `cann/pypto`（upstream） | 错误 | `git remote set-url origin https://gitcode.com/<username>/pypto.git` |
| 无 origin 或非 pypto | 错误 | 询问用户正确的仓库路径 |
