# 通用 Review 评论修复指南

本文档定义通用的 review 评论理解和修复策略。不预设固定分类，由 LLM 动态判断修复方案。

## 评论理解流程

```
1. 阅读评论全文
   ↓
2. 识别意图（修改请求 / 疑问 / 建议 / 拒绝）
   ↓
3. 判断可自动修复 vs 需人工判断
   ↓
4. 定位受影响文件
   ↓
5. 执行修复或生成 TODO
```

## 可自动修复的判断标准

满足**全部**条件时可自动修复：

1. **修改意图明确** — 评论清楚描述了需要什么改动
2. **范围可界定** — 能确定影响哪些文件和代码位置
3. **操作确定性强** — 修改方案唯一或选项有限
4. **无业务判断** — 不涉及设计决策、架构选择、业务逻辑权衡

### 典型可自动修复场景

- 移动/重命名字段或变量
- 修复格式问题（缩进、空格、换行）
- 添加缺失的必要内容（导入语句、类型注解、文档字符串）
- 替换过时的 API 调用
- 修复拼写错误
- 调整代码风格以符合项目规范

### 典型需人工判断场景

- "这个设计是否合理？"
- "建议考虑使用 X 方案替代"
- "这里的性能可能有问题"
- "需要增加测试覆盖"（需理解业务逻辑才能写测试）
- "安全风险：..."（需评估实际影响）

## 文件定位策略

### 优先级 1: 评论直接关联

`diff_comment` 类型包含 `diff_position`：

```json
{
  "diff_position": {
    "start_new_line": 6,
    "end_new_line": 6
  }
}
```

**注意**：`diff_position` 不包含文件 `path`。需结合评论中提及的代码片段通过 grep 定位。

### 优先级 2: 评论内容提取

从评论文本中提取文件路径、函数名、类名等关键信息：

```python
# 评论: "在 src/utils/helper.py 的 parse_config 函数中..."
# → 直接定位到 src/utils/helper.py

# 评论: "source_url 字段应该放在 metadata 下"
# → grep "source_url:" 定位包含该字段的文件
```

### 优先级 3: grep 搜索

当评论未明确指出文件时，根据评论提到的代码内容搜索：

```bash
# 搜索评论提到的具体代码
grep -rn "target_content" path/to/repo/
```

## 修复执行原则

1. **最小改动** — 只修改 reviewer 要求修改的部分，不做额外重构
2. **保留格式** — 保持文件原有的缩进风格、换行习惯
3. **可回滚** — 所有修改可通过 `git checkout` 恢复
4. **先验证后提交** — 每次修复后验证文件完整性

## GitCode 评论结构参考

### comment_type 字段

| comment_type | 含义 | 特征 |
|-------------|------|------|
| `pr_comment` | PR 级别评论 | 不关联具体代码行 |
| `diff_comment` | Diff 行内评论 | 关联代码变更，含 `diff_position` |

### 机器人过滤规则

| 优先级 | 过滤条件 | 示例 |
|--------|---------|------|
| 1 | `user.login == "cann-robot"` | CANN CI 机器人 |
| 2 | login 含 `bot`/`robot`/`ci` | 通用机器人 |
| 3 | body 以固定模板开头 | CI 状态报告 |

## 输出规范

### 自动修复报告

```yaml
auto_fixes:
  - comment_id: 164054117
    intent: "简明描述 reviewer 意图"
    files_changed:
      - path: "path/to/file.py"
        changes: "具体修改说明"
    status: success

manual_todos:
  - comment_id: 164054200
    intent: "简明描述评论内容"
    reason: "为什么无法自动修复"
    suggested_action: "建议的处理方式"
```
