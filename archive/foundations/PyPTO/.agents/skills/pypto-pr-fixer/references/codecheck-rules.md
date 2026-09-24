# CodeArts-Check 规则参考 (PyPTO CodeCheck)

PyPTO 仓库使用华为 CodeArts-Check 作为 CI 静态分析工具。当 cann-robot 在 PR 评论中报告 `codecheck ❌ FAILED` 时，本文档提供规则映射和自动修复指引。

> 规则的完整定义以官方 Excel 为准；建议使用 `scripts/query_codecheck_rule.py` 按需/批量查询，本文档更偏向“如何修”与“是否可自动修”。

## CI 评论格式

cann-robot 以 HTML 表格形式发布 CI 结果：

```html
<tr>
  <td><strong>codecheck</strong></td>
  <td>❌ FAILED</td>
  <td><a href="https://www.openlibing.com/apps/entryCheckDashCode/{MR_ID}/{hash}?projectId=300033&codeHostingPlatformFlag=gitcode">>>>>>></a></td>
</tr>
```

状态值：`❌ FAILED`、`✅ SUCCESS`、`⚪ ABORTED`

## 报告获取

openlibing.com 是 SPA 页面，受 WAF 保护，必须通过 Playwright 浏览器渲染获取完整报告内容。

### 方法一：使用提取脚本（推荐）

```bash
# 使用内置脚本提取违规列表
python scripts/fetch_codecheck_violations.py \
  "https://www.openlibing.com/apps/entryCheckDashCode/{MR_ID}/{hash}?projectId=300033" \
  --output json
```

输出格式：
```json
{
  "total": 79,
  "by_rule": {"G.LOG.02": 25, "G.FMT.02": 22, ...},
  "violations": [
    {
      "file": "path/to/file.py",
      "line": 42,
      "description": "Line too long (149/120)",
      "rule_id": "G.FMT.02",
      "rule_description": "行宽不超过 120 个字符"
    },
    ...
  ]
}
```

### 规则批量查询（推荐）

一次 codecheck 往往会暴露多个 `rule_id`。不建议逐条手动查本文档；可直接对违规提取的 JSON 批量查询官方定义：

```bash
python scripts/query_codecheck_rule.py \
  --from-violations-json violations.json \
  --language python \
  --format markdown
```

如果希望在推送前先做本地筛查，可运行：

```bash
python scripts/local_codecheck.py <repo_path> --output json
```

### 方法二：Playwright Python 直接调用

```python
from playwright.sync_api import sync_playwright
import re

def fetch_codecheck_violations(url: str) -> list[dict]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(3000)

        # 移除 cookie 对话框（会阻挡分页控件）
        page.evaluate("""
            const cookieDivs = document.querySelectorAll('[class*="cookie"]');
            cookieDivs.forEach(div => div.remove());
        """)
        page.wait_for_timeout(500)

        # 切换到最大页面大小以显示所有违规
        try:
            page.locator(".el-pagination__sizes").click(timeout=5000)
            page.wait_for_timeout(500)
            options = page.locator(".el-select-dropdown__item").all()
            if options:
                # 找最大数字选项并点击
                largest = max(options, key=lambda o: int(''.join(filter(str.isdigit, o.inner_text()))) or 0)
                largest.click()
                page.wait_for_timeout(2000)
        except:
            pass

        text = page.inner_text("body")
        browser.close()

        # 用正则解析违规列表
        pattern = r'文件路径:([^\n:]+):(\d+)\s*问题描述[：:]([^\n]+)\s*规则[：:]([^\n]+)'
        violations = []
        for match in re.findall(pattern, text):
            file_path, line, desc, rule = match
            rule_parts = rule.strip().split(" ", 1)
            violations.append({
                "file": file_path.strip(),
                "line": int(line),
                "description": desc.strip(),
                "rule_id": rule_parts[0],
                "rule_description": rule_parts[1] if len(rule_parts) > 1 else ""
            })
        return violations
```

### 注意事项

1. **ARM64 环境**：Playwright MCP 不支持 ARM64，必须使用 Playwright Python 库
2. **Cookie 对话框**：必须移除，否则会阻挡分页控件
3. **分页**：默认只显示 20 条，需切换到 100 条/页
4. **超时**：SPA 渲染需要时间，设置 60s 超时

## Python 规则总览

共 111 条规则，按类别分组。严重等级：critical > major > minor > suggestion。

### 可自动修复的规则（可自动修复）

以下规则具有确定性修复方案，适合自动修复：

#### 命名规范 (G.NAM)

| 规则 | 严重等级 | 说明 | 自动修复方案 |
|------|---------|------|-------------|
| G.NAM.01 | minor | 命名风格一致性 | 按 PEP 8 规范重命名 |
| G.NAM.03 | minor | 实例方法首参数用 `self`，类方法用 `cls` | 修正首参数名 |

#### 格式规范 (G.FMT) — 高频可自动修复

| 规则 | 严重等级 | 说明 | 自动修复方案 |
|------|---------|------|-------------|
| G.FMT.01 | minor | 代码块缩进 4 空格 | 调整缩进 |
| G.FMT.02 | minor | 每行最多 120 字符 | 拆分长行 |
| G.FMT.03 | minor | 合理使用空行（类间 2 行，方法间 1 行，嵌套定义 1 行） | 增删空行 |
| G.FMT.04 | minor | 关键字和运算符前后加空格 | 添加/删除空格 |
| G.FMT.05 | minor | import 放在模块注释和 docstring 之后 | 调整 import 位置 |
| G.FMT.06 | minor | 每行只导入一个模块 | 拆分 import 语句 |
| G.FMT.07 | minor | import 按标准库、第三方、本地分组 | 重排 import 顺序 |
| G.FMT.08 | minor | 每行只写一条语句 | 拆分语句 |

#### 注释规范 (G.CMT)

| 规则 | 严重等级 | 说明 | 自动修复方案 |
|------|---------|------|-------------|
| G.CMT.01 | minor | 模块 docstring 位置 | 调整位置 |
| G.CMT.04 | minor | 注释位置和格式一致 | 调整格式 |
| G.CMT.05 | minor | 避免 TODO/FIXME 注释 | 删除或转为 issue |
| G.CMT.06 | minor | 文件头必须包含版权声明 | **重要**：文件头必须遵循 PyPTO 仓库其他文件的格式——不仅仅是版权声明，还包括完整的文件头结构。请参考仓库中现有文件获取正确格式。 |

#### 运算符规范 (G.OPR)

| 规则 | 严重等级 | 说明 | 自动修复方案 |
|------|---------|------|-------------|
| G.OPR.02 | minor | 与 None 比较用 `is`/`is not` | 替换 `==`/`!=` |
| G.OPR.03 | major | 内置类型比较不用 `is`/`is not` | 替换为 `==`/`!=` |
| G.OPR.05 | major | 用 `is not` 而非 `not ... is` | 语法改写 |
| G.OPR.06 | minor | 用 `not in` 测试成员关系 | 语法改写 |

#### 表达式规范 (G.EXP)

| 规则 | 严重等级 | 说明 | 自动修复方案 |
|------|---------|------|-------------|
| G.EXP.03 | minor | lambda 不应赋值给变量，用 `def` | 改写为 def |

#### 控制流 (G.CTL)

| 规则 | 严重等级 | 说明 | 自动修复方案 |
|------|---------|------|-------------|
| G.CTL.05 | minor | 循环中未使用变量用 `_` | 替换变量名为 `_` |

#### 类型相关 (G.TYP)

| 规则 | 严重等级 | 说明 | 自动修复方案 |
|------|---------|------|-------------|
| G.TYP.04 | minor | 用 `if seq` / `if not seq` 判断空序列 | 简化条件表达式 |
| G.TYP.08 | major | 用 `isinstance()` 判断类型 | 替换 `type()` 比较 |

#### import 规范 (G.IMP)

| 规则 | 严重等级 | 说明 | 自动修复方案 |
|------|---------|------|-------------|
| G.IMP.01 | minor | 使用绝对导入 | 改写为绝对路径 |
| G.IMP.02 | minor | `from ... import ...` 注意事项 | 避免通配符导入 |

#### 错误处理 (G.ERR)

| 规则 | 严重等级 | 说明 | 自动修复方案 |
|------|---------|------|-------------|
| G.ERR.04 | major | 异常类型转换时保留原始调用栈 | 使用 `raise ... from e` |
| G.ERR.06 | minor | raise 必须包含异常实例 | 补全异常实例 |
| G.ERR.07 | minor | 避免忽略异常 | 添加日志或重新抛出 |
| G.ERR.13 | minor | 避免直接重新抛出已捕获的异常 | 使用 `raise` 不带参数 |

#### 项目规范 (G.PRJ)

| 规则 | 严重等级 | 说明 | 自动修复方案 |
|------|---------|------|-------------|
| G.PRJ.03 | **critical** | 代码中不能包含调试入口点 | 删除 `pdb`/`breakpoint()` |
| G.PRJ.04 | minor | 项目统一使用 UTF-8 编码 | 添加编码声明 |
| G.PRJ.05 | major | 删除无用代码而非注释掉 | 删除注释掉的代码块 |
| G.PRJ.06 | major | 删除正式代码中的个人信息 | 移除个人信息 |

#### 日志规范 (G.LOG)

| 规则 | 严重等级 | 说明 | 自动修复方案 |
|------|---------|------|-------------|
| G.LOG.02 | minor | 使用日志记录工具实现日志功能 | 替换 `print` 为 `logging` |

### 需要人工判断的规则（需人工判断）

以下规则修复涉及业务逻辑理解，建议生成修复建议但需人工确认：

#### 函数规范 (G.FNM)

| 规则 | 严重等级 | 说明 |
|------|---------|------|
| G.FNM.01 | major | 不使用可变对象作为默认参数值 |
| G.FNM.02 | major | 函数/lambda 不应使用外层循环变量 |
| G.FNM.04 | major | 不将无返回值的函数结果赋给变量 |

#### 控制流 (G.CTL)

| 规则 | 严重等级 | 说明 |
|------|---------|------|
| G.CTL.01 | major | 函数所有分支的返回值类型和数量一致 |
| G.CTL.02 | major | 所有代码必须逻辑可达 |

#### 类规范 (G.CLS)

| 规则 | 严重等级 | 说明 |
|------|---------|------|
| G.CLS.01 | major | 子类 `__init__` 需正确调用父类 `__init__` |
| G.CLS.06 | major | 类的方法建议统一按照一种规则进行排列 |
| G.CLS.09 | major | 重写魔术方法需返回规定类型 |
| G.CLS.10 | major | 数值运算魔术方法返回 `NotImplemented` 而非抛 `NotImplementedError` |
| G.CLS.11 | major | 避免在类外或者子类中访问父类受保护的成员 |

#### 安全规范 (G.SER/G.EDV/G.DSP)

| 规则 | 严重等级 | 说明 |
|------|---------|------|
| G.SER.03 | major | 不使用 YAML 的 `load()` 函数 |
| G.EDV.03 | **critical** | 命令解析器避免通配符 |
| G.EDV.04 | major | subprocess 不使用 `shell=True` |
| G.EDV.05 | major | 调用外部可执行程序时建议使用绝对路径 |
| G.FIO.01 | major | 创建文件时指定适当权限 |
| G.DSP.02 | major | 安全场景使用加密安全随机数 |

### 其他规则

完整规则定义以官方 Excel 为准（`rule_ch.xlsx` / `rule_en.xlsx`），建议使用 `scripts/query_codecheck_rule.py` 查询。

## 修复流程

1. **解析报告**：使用上述方法从 openlibing.com 报告中提取违规列表（规则ID + 文件 + 行号）
2. **匹配规则**：根据规则 ID 在本文档中查找修复方案
3. **分类处理**：
   - 可自动修复 → 直接应用修复
   - 需人工判断 → 生成修复建议，标记为 TODO
4. **验证修复**：修复后重新运行 codecheck（若可行）或通过代码审查确认

## 规则来源

- **仓库**：`https://gitcode.com/cann/infrastructure/tree/main/docs/SC/CodeArts-Check`
- **Excel**：`rule_en.xlsx`（英文）、`rule_ch.xlsx`（中文）
- **VS Code 插件**：`https://marketplace.visualstudio.com/items?itemName=HuaweiCloud.codecheck`
