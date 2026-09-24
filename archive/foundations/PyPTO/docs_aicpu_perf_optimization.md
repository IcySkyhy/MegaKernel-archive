# 昇腾 AICPU 侧高性能编码指南

> **权威位置（skill）**：[`.agents/skills/pypto-aicpu-perf-coding/`](.agents/skills/pypto-aicpu-perf-coding/SKILL.md)
> 完整原则：[`references/coding-principles.md`](.agents/skills/pypto-aicpu-perf-coding/references/coding-principles.md)
> 本文件为仓库根目录入口副本，修改时请与 skill 内文档同步。

Agent 在修改 / 评审 AICPU 热路径时应加载 skill `pypto-aicpu-perf-coding`。

---

其余内容见 skill 引用文档（禁 STL 容器、平凡默构、忌整结构初始化、惰性 bump、flat 小池、hit 零拷贝等）。
