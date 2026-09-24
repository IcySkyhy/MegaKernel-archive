# AGENTS.md

## 项目概述

本仓库为华为昇腾 AI 处理器高性能编程框架 **PyPTO（CANN）** 的源码仓。此处的 agent 与 skill 面向 **PyPTO 框架自身的开发与维护**，覆盖框架调试与错误定位、编译期 Pass 分析、环境配置、以及 PR / Issue 流程。

### 核心功能

- PyPTO 框架的调试、错误定位与 Pass 模块分析
- 环境安装与依赖问题诊断
- 规范化的 PR / Issue 创建与修复流程

---

## 通用原则

> **严格遵循以下原则**

1. **编码前思考** —— 不要假设，不要隐藏困惑，呈现权衡
   - 明确说明假设——如果不确定，询问而不是猜测
   - 存在歧义时呈现多种解释，不要默默选择
   - 如果存在更简单的方法，适时提出异议
   - 困惑时停下来，指出不清楚的地方并要求澄清
2. **简洁优先** —— 用最少的代码解决问题，不要过度推测
   - 不要添加要求之外的功能；不要为一次性代码创建抽象
   - 不要添加未要求的"灵活性"或"可配置性"；不要为不可能发生的场景做错误处理
   - 检验标准：资深工程师会觉得这过于复杂吗？如果是，简化
3. **精准修改** —— 只碰必须碰的，只清理自己造成的混乱
   - 不要"改进"相邻的代码、注释或格式；不要重构没坏的东西；匹配现有风格
   - 如果注意到无关的死代码，提一下，不要删除它
   - 只删除因自己改动而变得无用的导入/变量/函数，不删除预先存在的死代码
   - 检验标准：每一行修改都应该能直接追溯到用户的请求
4. **目标驱动执行** —— 定义成功标准，循环验证直到达成
   - "添加验证"→"为无效输入编写测试，然后让它们通过"；"修复 bug"→"编写重现 bug 的测试，然后让它通过"
   - 多步骤任务先说明简短计划（每步 → 验证）
5. **先验证，再下结论** —— 结论必须有证据支撑，不能靠猜
   - 代码、文档、日志能查证的，不要凭记忆或直觉判断
   - 明确区分已确认的事实与你的推测，不要把推测当结论说
   - 检验标准：结论要能说清依据来源，经得起追问

---

## 入口路径速查

| 你想做什么 | 第一站 |
|---|---|
| 定位 AICore / Host 错误 | skill `pypto-aicore-error-locator` / `pypto-host-stacktrace-analyzer` |
| 分析编译期 Pass 模块 | skill `pypto-pass-module-analyzer` / `pypto-pass-workflow-analyzer` |
| 排查 workspace 内存问题 | skill `pypto-memory-overlap-detector` / `pypto-machine-workspace` |
| 调试算子精度问题 | skill `pypto-precision-overall` |
| AICPU / CF 热路径高性能编码与评审 | skill `pypto-machine-code-review` |
| 安装 / 修复环境 | skill `pypto-environment-setup` |
| 创建 / 修复 PR、创建 Issue | skill `pypto-pr-creator` / `pypto-pr-fixer` / `pypto-issue-creator` |
| 检视 machine 侧代码、提交行级 review 意见 | skill `pypto-machine-code-review` |
