# PyPTO PR 提交前检查清单

> 阶段 4（预检）和阶段 6（创建 PR）前加载，逐项检查。

## 环境预检

- [ ] 本地仓库路径正确
- [ ] origin 指向用户 fork（非 `cann/pypto`）
- [ ] upstream remote 已添加：`git remote add upstream https://gitcode.com/cann/pypto.git`（如不存在）
- [ ] fork 链关系已验证（`parent.full_name == "cann/pypto"`）
- [ ] 浅克隆已修复（如需要）：`git fetch --unshallow origin`

## Git 认证

- [ ] 认证方式已配置（cache/store/SSH/Token/libsecret）
- [ ] `git push --dry-run origin <branch>` 验证通过

## 代码准备

- [ ] upstream 已同步：`git fetch upstream master`
- [ ] 分支未落后 upstream：`git log --oneline HEAD..upstream/master` 输出为空
- [ ] feat/fix 类型已添加测试用例
- [ ] code-check 警告已修复（参考 `docs/zh/contribute/code-check-rule.yaml`）

## Commit 规范

- [ ] 格式：`tag(scope): Summary`
- [ ] Tag 为合法类型：feat / fix / docs / style / refactor / test / perf
- [ ] Summary 英文、首字母大写、无句号、祈使语气、10-200 字符
- [ ] 整个 commit message 不超过 10 行
- [ ] 每个 commit 只做一件事

## PR 规范

- [ ] 标题遵循 `tag(scope): Summary` 格式
- [ ] Body 清晰描述变更意图（禁止为空）
- [ ] 单一职责 — 无不相关变更混入
- [ ] 已关联相关 Issue

## Cross-Fork 参数

- [ ] `head` 使用 `<username>:<branch_name>` 格式（冒号分隔）
- [ ] MCP 参数名全小写
- [ ] `owner`/`repo` 指向上游仓库 `cann/pypto`

## 用户确认

- [ ] 已向用户展示完整执行计划表
- [ ] 用户已明确确认
