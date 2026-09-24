---
name: pypto-issue-creator
description: 基于会话上下文智能创建 GitCode Issue。支持 5 种类型：Bug Report（报错/异常/失败/精度问题）、Feature Request（新功能/优化）、Documentation（文档问题）、Question（使用问题）、Task（开发任务）。触发词：创建issue、提交issue、反馈问题、报告bug、功能请求、文档问题、提问、咨询、创建任务、跟踪任务。
---

# PyPTO Issue Creator

基于会话上下文，智能创建符合 PyPTO 和 CANN 社区规范的 GitCode Issue。

## 环境依赖

### GitCode MCP

本 skill 的去重检查（阶段3）和 Issue 创建（阶段8）依赖 GitCode MCP 工具。

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

## 目录

1. [工作流程](#工作流程)
2. [标题格式规范](#标题格式规范)
3. [Issue 类型识别](#issue-类型识别)
4. [去重检查](#去重检查)
5. [环境信息获取](#环境信息获取)
6. [Issue 内容生成](#issue-内容生成)
7. [脚本与参考文档](#脚本与参考文档)
8. [执行检查清单](#执行检查清单)

---

## 工作流程

执行以下 8 阶段流程：

```
阶段1: 上下文理解
    ↓
阶段2: 类型识别与推断
    ↓
阶段3: 去重检查 (GitCode MCP)
    ↓
阶段4: 类型化验证 + 环境信息获取
    ├─ 4a: Bug Report → 自动验证 + 完整环境
    ├─ 4b: Feature Request → 文档/代码/PR/生态对比
    ├─ 4c: Question/Documentation → 文档验证
    └─ 4d: Task → 确认完整
    ↓
阶段5: 智能交互 (补充缺失信息)
    ↓
阶段6: Issue 内容生成
    ↓
阶段7: 最终确认与创建模式
    ├─ 默认远程创建 → 使用 gitcode_create_issue（用户未明确指定本地时）
    └─ 用户明确指定本地 → 保存为本地 markdown
        ├─ 确定保存路径（默认: ./issues/）
        ├─ 文件名格式: issue-{type}-{YYYYMMDDHHmmss}.md
        └─ 提示: "Issue 已保存至本地 {路径}，未提交远程"
    ↓
阶段8: 创建 Issue
```

---

## 标题格式规范

所有 Issue 标题必须遵循以下格式：

```
[英文类型|中文类型]: 具体描述
```

仅允许以下 5 种前缀：

| 前缀 | 对应类型 |
|------|---------|
| `[Bug-Report\|缺陷反馈]` | Bug Report |
| `[Requirement\|需求建议]` | Feature Request |
| `[Documentation\|文档反馈]` | Documentation |
| `[Question\|问题咨询]` | Question |
| `[Task\|任务跟踪]` | Task |

### 示例

| 正确 ✅ | 错误 ❌ |
|--------|--------|
| `[Bug-Report\|缺陷反馈]: 执行测试用例aicpu超时` | `[bug][needs-triage] 执行测试用例aicpu超时` |
| `[Task\|任务跟踪]: 分布式任务适配fixed_output_path` | `[Task] 分布式任务适配fixed_output_path` |
| `[Requirement\|需求建议]: 支持动态shape的算子开发` | `[Feature-Request\|功能请求]: 支持动态shape的算子开发` |
| `[Question\|问题咨询]: 如何跑framework用例` | `如何跑framework用例` |
| `[Documentation\|文档反馈]: view文档中viewOp说明缺失` | `[docs] view文档中viewOp说明缺失` |

### 规则

1. 使用方括号 `[]` 包裹类型标签
2. 英文类型与中文类型之间用管道符 `|` 分隔
3. 类型标签后使用冒号 `: ` 和空格分隔具体描述
4. GitCode Label 按项目特性模块设置（如 Frontend、Operator、Passes 等），由 Maintainer/Committer 在分发阶段添加，不在标题中体现

---

## Issue 类型识别

根据会话上下文自动判断类型：

| 类型 | 触发信号 | 必需信息 | 验证要求 |
|-----|---------|---------|---------|
| **Bug Report** | 报错、异常、失败、精度问题 | 问题描述、环境信息、重现步骤、预期结果、日志/截图 | 自动验证 + 完整环境 |
| **Feature Request** | 希望支持、新增功能、优化、扩展 | 背景信息、信息来源 | 文档/代码/PR/生态对比 |
| **Documentation** | 文档缺失、文档错误、文档改进 | 文档链接、问题文档片段 | 文档验证 |
| **Question** | 如何使用、为什么、不理解 | 问题描述 | 文档验证 |
| **Task** | 需要开发、计划、任务 | 任务描述、任务目标 | 确认完整 |

---

## 去重检查

**阶段 3 执行**，在生成 Issue 内容前必须检查：

### 1. GitCode 远程检查

```bash
# 关键词搜索
gitcode_search_issues(query="repo:cann/pypto {关键词}")
```

### 2. 标题前缀搜索

```bash
# 按标题前缀类型搜索，提高去重精度
gitcode_search_issues(query="repo:cann/pypto [Bug-Report|缺陷反馈] {关键词}")
gitcode_search_issues(query="repo:cann/pypto [Task|任务跟踪] {关键词}")
```

### 3. 中英文双语搜索

```bash
# 中文和英文关键词分别搜索
gitcode_search_issues(query="repo:cann/pypto {中文关键词}")
gitcode_search_issues(query="repo:cann/pypto {english keywords}")
```

### 4. 处理策略

- **发现重复**: 展示已有 Issue（含编号、标题、状态），询问用户是否继续
- **无重复**: 继续下一阶段

---

## 环境信息获取

**Bug Report 类型需要获取完整环境信息**。其他类型 Issue 可根据需要选择性获取。

运行采集脚本自动获取：
```bash
bash scripts/collect-env.sh
```

### 快速参考

| 信息类型 | 命令 | 输出示例 |
|---------|------|---------|
| 服务器/NPU 型号 | `lspci -n -D \| grep -oE '19e5:d80[23]'` | `19e5:d803` |
| CANN 版本 | `echo $ASCEND_HOME_PATH \| sed -n 's/.*cann-\([0-9.]*\).*/\1/p'` | `9.0.0` |
| PyPTO Commit | `git log -1 --format='%h (%ci)' HEAD` | `abc1234 (2026-03-10 10:00:00 +0800)` |
| Python 版本 | `python --version` | `Python 3.10.12` |
| 操作系统 | `grep '^PRETTY_NAME=' /etc/os-release \| cut -d'=' -f2- \| tr -d '"'` | `Ubuntu 22.04.3 LTS` |
| torch 版本 | `python -c "import torch; print(torch.__version__)"` | `2.6.0` |
| torch_npu 版本 | `python -c "import torch_npu; print(torch_npu.__version__)"` | `2.6.0.post3` |

---

## Issue 内容生成

### 模板选择

根据类型加载对应模板（详见 [references/issue-templates.md](references/issue-templates.md)）：

- **Bug Report**: 问题描述 → 环境信息 → 重现步骤 → 预期结果 → 日志截图 → 备注
- **Feature Request**: 背景信息 → 信息来源 → 价值/作用 → 设计方案
- **Documentation**: 文档链接 → 问题文档片段 → 存在的问题 → 修正建议
- **Question**: 问题描述 → 已尝试的方法
- **Task**: 任务描述 → 任务目标

---

## 脚本与参考文档

### references/

| 文件 | 内容 |
|-----|------|
| [issue-templates.md](references/issue-templates.md) | 5 种 Issue 类型的完整模板 |
| [collect-env.sh](scripts/collect-env.sh) | 环境信息自动采集脚本 |

---

## 执行检查清单

创建 Issue 前确认：

- [ ] 类型判断正确
- [ ] 标题格式符合 `[英文类型|中文类型]: 具体描述` 规范
- [ ] 去重检查完成
- [ ] 必需信息完整
- [ ] 环境信息已获取（Bug Report）
- [ ] 信息来源已填写（Feature Request）
- [ ] 内容符合模板规范（中英双语字段名、Mandatory/Optional 标注）
- [ ] 创建模式已确认（未明确指定本地时默认远程创建）
- [ ] 用户已确认
