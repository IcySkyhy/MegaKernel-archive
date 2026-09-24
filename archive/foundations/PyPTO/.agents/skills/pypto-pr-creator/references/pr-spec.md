# PyPTO PR 规范

> 来源：GitCode cann/pypto 仓库 `docs/zh/contribute/pull-request.md`。编写 commit message 和 PR 内容时加载。

## Commit Message 格式

```
tag(scope): Summary
```

### Tag 类型

| Tag | 用途 |
|-----|------|
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `docs` | 文档变更 |
| `style` | 代码格式 |
| `refactor` | 重构 |
| `test` | 测试相关 |
| `perf` | 性能优化 |

> ⚠️ `chore` 不在允许列表中。

### Scope 规则

- 填写受影响的模块/组件名称
- 多模块用 `|` 分隔：`feat(frontend|backend): ...`
- 涉及过多模块时使用 `all`：`refactor(all): ...`

### Summary 规则

- 英文编写，首字母大写，结尾不加句号，祈使语气
- 长度 10-200 字符
- 正则验证：`^(feat|fix|docs|style|refactor|perf|test)(.*): [A-Z].{10,200}`

### 示例

| ✅ 正确 | ❌ 错误 | 原因 |
|---------|---------|------|
| `Add support for FP16 data type` | `Added support for FP16` | 过去时态 |
| `Fix memory leak in graph optimization` | `Fixes memory leak in optimizer.` | 祈使句尾不加句号 |
| `Update tensor creation API documentation` | `updated API documentation` | 未首字母大写 |
| `feat(skills): Add PR creator skill` | `feat(skills): 为 PyPTO 项目添加 PR 创建技能` | 中文 |
| `fix(ops): Fix precision issue in softmax` | `feat: add feature` | 无 scope |

### 整体要求

- 每个 commit 只做一件事
- 多 commit 推荐顺序：`fixup` → `refactor` → `feat` → `test`
- 整个 commit message 不超过 10 行
- 必须使用英文

---

## PR 标题格式

与 Commit Message 格式相同：`tag(scope): Summary`。

## PR Body 格式

```
tag(scope): Summary

[变更描述 - 动机、背景、上下文]

Changes:
- [变更点1]
- [变更点2]

Related Issues: #1234,#5678
```

### 要求

1. 禁止空 Body — 必须清晰传达变更意图
2. 描述动机和上下文 — 让 reviewer 理解 why
3. 移除无关模板内容和占位符
4. 关联 Issue 放在最后一行 `Related Issues: #xxx`
5. 多 commit 场景需在 Body 中列出每个 commit 摘要

### 完整示例

```
feat(interface): Optimize the pypto.cond with concrete value

The origin implementation of pypto.cond generate both if/else branch,
even if the condition is always true or false, in this PR, we optimize
the implementation to generate only one branch.

Changes:
- Optimize the pypto.cond with concrete value, which can reduce the number of branches in the program
- Update the test cases to cover the new features

Related Issues: #1234,#5678
```
