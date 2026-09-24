---
name: gitcode-mcp-install
description: 安装和配置 GitCode MCP Server，使 AI 客户端能与 GitCode 平台交互（仓库/分支/Issue/PR 管理）。触发词：安装 gitcode mcp、配置 gitcode mcp、gitcode mcp server。
---


# GitCode MCP Server 安装与配置

## ⛔ 隐私保护

> ⚠️ **禁止在屏幕、日志、错误信息中打印 `GITCODE_TOKEN` 环境变量**

Token 仅存储在安全位置（环境变量或配置文件），不要在终端输出中暴露。文档/代码示例必须用占位符 `<YOUR_GITCODE_TOKEN>`。


## 安装

两种安装方式按环境选择。

### 方式一：Go 二进制

```bash
go install gitcode.com/gitcode-ai/gitcode_mcp_server@latest
```

> **注意**：Go 二进制方式在国内网络可能失败（GitCode 不支持 Go module 代理）。
> 若失败请使用 Python 方式。

### 方式二：Python 源码安装（推荐）

**标准安装**：
```bash
git clone https://gitcode.com/gitcode-ai/gitcode_mcp_server.git /tmp/gitcode_mcp_server
pip3 install -e /tmp/gitcode_mcp_server
```

**安装 PR #3 版本（推荐，修复分页截断问题）**：

> PR #3 修复了 `list_pull_request_comments` 分页截断问题，新增评论类型过滤、回复评论等功能。

```bash
# 克隆仓库
git clone https://gitcode.com/gitcode-ai/gitcode_mcp_server.git /tmp/gitcode_mcp_server
cd /tmp/gitcode_mcp_server

# 获取并切换到 PR #3
git fetch origin +refs/merge-requests/3/head:pr_3
git checkout pr_3

# 安装
pip3 install -e .
```

> Python >= 3.8
>
> **重要**：Python 安装方式会自动注册 `gitcode-mcp` 命令，配置方式与 Go 二进制相同。
## OpenCode 配置（不存在会自动创建）

脚本会在 `~/.config/opencode/opencode.json` 不存在时创建默认配置，并把 token 写成占位符。
你只需要在安装完成后把占位符替换成真实 token，**修改后需重启 OpenCode 才能生效**。

模板如下（按需求固定格式）：

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "gitcode": {
      "type": "local",
      "command": [
        "gitcode-mcp"
      ],
      "enabled": true,
      "environment": {
        "GITCODE_TOKEN": "<YOUR_GITCODE_TOKEN>",
        "GITCODE_API_URL": "https://api.gitcode.com/api/v5"
      }
    }
  }
}
```

注意：不要在对话里发送或者询问GITCODE_TOKEN，若用户主动提供需要提醒用户存在泄露风险。只需要修改本机文件 `~/.config/opencode/opencode.json`。

## 获取 Token（不在这里索取）

1. 登录 https://gitcode.com → 设置 → 访问令牌
2. 创建 Personal Access Token（建议包含 `repo`、`read:user` 权限）
3. 保存 Token（仅显示一次）

## 验证

**手动验证**：

```bash
# 1. 检查命令是否存在
which gitcode-mcp

# 2. 检查配置文件
cat ~/.config/opencode/opencode.json | grep -A5 gitcode

# 3. 测试 API 连接（替换 <YOUR_TOKEN> 为真实 token）
curl -s "https://api.gitcode.com/api/v5/user/repos?access_token=<YOUR_TOKEN>&per_page=5" | jq '.[].full_name'
```

**验证要点**：
- `which gitcode-mcp` 返回路径
- 配置文件中 `GITCODE_TOKEN` 不是占位符
- API 调用返回仓库名称列表

> **注意**：修改 `~/.config/opencode/opencode.json` 后需重启 OpenCode 才能生效。

## 代理（可选）

如需代理访问 GitCode API，把代理环境变量加入 OpenCode 的 `environment`：

```json
"environment": {
  "GITCODE_TOKEN": "<YOUR_GITCODE_TOKEN>",
  "GITCODE_API_URL": "https://api.gitcode.com/api/v5",
  "HTTP_PROXY": "http://proxy:8080",
  "HTTPS_PROXY": "http://proxy:8080"
}
```

## 故障排查

| 问题 | 处理 |
|------|------|
| `gitcode-mcp: command not found` | 确认 PATH：`which gitcode-mcp` |
| API 401/403 | token 无效/权限不足，更新 `~/.config/opencode/opencode.json` |
| 连接超时 | 检查网络或配置代理 |
| 配置文件不存在 | 手动创建 `~/.config/opencode/opencode.json` |

## 外部参考

- 源码: https://gitcode.com/gitcode-ai/gitcode_mcp_server
- MCP 协议: https://modelcontextprotocol.io
- GitCode API 文档: https://docs.gitcode.com/docs/apis/
