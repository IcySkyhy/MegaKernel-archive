# PyPTO Pass Bug Pattern Catalog

本目录记录从 PyPTO 仓库历史 `fix(pass)` / `feat(pass)` 提交中沉淀出的常见 bug 模式。它是一个**静态知识库**，用于在 agent 遇到 Pass 相关失败时快速给出定界假设和验证方向。

> 说明：本目录不执行在线 git 检索，也不自动更新。`../patterns/` 是结构化唯一事实源；本目录只提供人类可读导航。修改结构化条目时，必须同步核对同名索引，避免两份内容漂移。

---

## 目录

1. [跨模块 Bug 定界](cross-module.md)
2. [Pass 历史模式](pass-patterns.md)
3. [Pass 源码审计模式](source-patterns.md)
4. [非 Pass 源码审计模式](non-pass-patterns.md)
5. [Pass 禁用诊断策略](disable-policy.md)

---

## 如何使用本目录

当 agent 接到 PyPTO 相关失败时，**不要一见 Pass 报错就只在 Pass 里找原因**。按以下顺序使用：

1. **提取现象关键词**：从报错日志、IR 差异、计算图变化、错误码中提炼 2-5 个关键词。
2. **先做跨模块定界**：根据 `../patterns/cross-module.json` 中的 `cross_module_triage` 判断最可能的模块（pass / operation / machine / interpreter / frontend / distributed / codegen）。
3. **再做 Pass 内模式匹配**：如果最可能模块是 pass，再匹配 `../patterns/pass-patterns.json` 中的 `patterns` 和 `../patterns/source-patterns.json` 中的 `source_patterns`。
4. **读取详细模式**：每个匹配到的模式给出：
   - 典型反模式（代码里容易错成什么样）
   - 修复模式（历史上怎么修）
   - 验证提示（应该重点查什么）
5. **形成根因假设**：不要直接当成结论，而是形成可验证的假设。

---

## 与现有技能的衔接

本目录本身只做**模式匹配和假设生成**，不做代码修改，也不做深度日志解析。匹配后应按下表衔接：

| 下一步 | 技能 | 目的 |
|--------|------|------|
| 获取日志、IR、计算图证据 | `pypto-pass-error-locator` | 验证假设、定位源码行 |
| 理解目标 Pass 的业务逻辑 | `pypto-pass-module-analyzer` | 确认反模式是否存在于当前代码 |
| 需要补充回归 UT | `pypto-pass-ut-generate` | 防止同样模式再次触发 |
| 涉及性能问题 | `pypto-pass-perf-optimizer` | 在功能修复后分析性能影响 |

---

## 维护说明

1. 新增模式时，先在 `../patterns/` 的对应 JSON 文件中添加结构化条目，再在同名 Markdown 文件中补充说明。
2. 历史 fix 模式应在 `example_commits` 中提供真实 commit；源码审计模式如果没有确认的历史提交，保留空数组，并在描述中保持“风险/假设”措辞。
3. Pass 模式保持 `phenomenon_keywords` 简洁，非 Pass 模式保持 `symptom_keywords` 简洁，优先使用报错日志中出现的高频词。
4. `verification_hints` 必须具体，不能写“检查代码”这类泛化描述。
5. 修改后人工核对 ID、提交/Issue、Pass 反向引用、依赖关系和本地路径；不得只更新 JSON 或只更新 Markdown。
