# GitCode PR Review Fixer 错误处理参考

## Section 1: GitCode MCP 错误类

| 错误类 | HTTP 状态码 | 触发场景 | 建议处理 |
|--------|-------------|----------|----------|
| `GitCodeAuthError` | 401 | Token 无效或过期 | 检查 GITCODE_TOKEN 环境变量 |
| `GitCodePermissionError` | 403 | 无仓库写入权限 | 检查 Token 权限范围 |
| `GitCodeNotFoundError` | 404 | PR/仓库不存在 | 检查 owner/repo/pull_number |
| `GitCodeValidationError` | 422 | 请求参数无效 | 检查参数格式 |
| `GitCodeRateLimitError` | 429 | API 调用频率超限 | 等待后重试 |
| `GitCodeServerError` | 500+ | GitCode 服务端错误 | 等待后重试 |
| `GitCodeAPIError` | 其他 | 通用 API 错误 | 查看错误详情 |

## Section 2: GitCode 平台错误

| error_code_name | error_message | 触发场景 | 诊断步骤 |
|-----------------|---------------|----------|----------|
| `BAD_REQUEST` | `Invalid header parameter: private-token, required` | curl 调用缺少 PRIVATE-TOKEN header | 添加 `-H "PRIVATE-TOKEN: ${GITCODE_TOKEN}"` |
| `UN_KNOW` | `pre receive hook check failed` | PR 创建时 commit 不符合服务端钩子检查 | 见 Section 3 |
| `UN_KNOW` (via MCP) | `API 调用失败: create_pull_request(...) - API 错误 (400): 未知错误` | MCP 内部封装失败，实际原因被吞 | 使用 curl 直接调用获取原始错误 |

## Section 3: pre-receive hook 诊断表

**说明**：pre-receive hooks 是 GitCode 平台级别的服务端钩子，不在 pypto 仓库中。

| 检查项 | 诊断命令 | 通过标准 | 修复建议 |
|--------|----------|----------|----------|
| Commit message 格式 | `git log -1 --format="%s"` | 必须匹配 `^(feat\|fix\|docs\|style\|refactor\|test\|perf)\(.+\): .+` | 重写 commit message |
| 分支同步状态 | `git log HEAD..origin/<target_branch> --oneline \| wc -l` | 应该为 0 | `git pull --rebase origin <target_branch>` |
| 文件大小限制 | `git diff --stat HEAD~1` | 单文件不超过 100MB | 拆分大文件 |
| 提交者身份 | `git log -1 --format="%ae"` | 邮箱格式合法 | 配置 git user.email |

## Section 4: MCP 工具可靠性参考

基于 PR 1276 实战经验：

| MCP 工具 | 可靠性 | 说明 |
|----------|--------|------|
| `gitcode_list_pull_request_comments` | 稳定 ✅ | 可直接使用 |
| `gitcode_get_pull_request` | 稳定 ✅ | 可直接使用 |
| `gitcode_create_pull_request` | 不稳定 ⚠️ | ~43% 成功率，建议使用 curl 备选 |
| `gitcode_create_pull_request_comment` | 未测试 | 需进一步验证 |

## Section 5: curl 备选方案模板

当 MCP 工具失败时，使用以下 curl 命令作为备选方案：

```bash
curl -X POST "https://api.gitcode.com/api/v5/repos/${OWNER}/${REPO}/pulls" \
  -H "Content-Type: application/json" \
  -H "PRIVATE-TOKEN: ${GITCODE_TOKEN}" \
  -d '{
    "title": "<title>",
    "head": "<source_branch>",
    "base": "<target_branch>",
    "body": "<description>"
  }'
```

## Section 6: 版本说明

- v1.1.0: 初始版本，基于 PR 1276 实战经验总结
