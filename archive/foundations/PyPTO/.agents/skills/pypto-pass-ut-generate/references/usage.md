# Pass UT 生成工具使用指南

## 概述

本工具用于为 PyPTO Pass 模块生成单元测试用例（UT），支持：
- 在线处理 PR，获取 diff 和覆盖率报告
- 离线分析本地 diff 文件和覆盖率报告
- 生成 UT 设计建议

## 工具目录结构

```
pypto-pass-ut-generate/
├── SKILL.md                  # 主技能文档
├── scripts/
│   ├── pr_utils.py           # PR 处理工具（在线）
│   ├── get_ut_status.py      # 快速获取 UT 状态
│   ├── ut_coverage.py        # 覆盖率分析工具
│   └── common_utils.py       # 公共工具函数
└── references/
    ├── check_list.md         # 检查清单
    └── usage.md              # 本文档
```

## 使用方式

### 方式一：在线处理 PR

```bash
# 处理 PR（自动获取 diff 和覆盖率报告）
python3 scripts/pr_utils.py 1894

# 快速获取 UT 状态
python3 scripts/get_ut_status.py 2017
```

### 方式二：离线分析本地 diff 文件

```bash
# 分析本地 diff 文件
python3 scripts/ut_coverage.py --diff /path/to/pr.diff

# 简短写法（文件在当前目录）
python3 scripts/ut_coverage.py --diff pr.diff
```

### 方式三：离线分析本地覆盖率报告

```bash
# 解析本地覆盖率报告（.tar.gz 格式）
python3 scripts/ut_coverage.py --report /path/to/ut_cov.tar.gz

# 简短写法（文件在当前目录）
python3 scripts/ut_coverage.py --report ut_cov.tar.gz
```

### 方式四：离线综合分析

```bash
# 同时分析 diff 和覆盖率文件
python3 scripts/ut_coverage.py --diff /path/to/pr.diff --report /path/to/ut_cov.tar.gz

# 输出 JSON 格式建议
python3 scripts/ut_coverage.py --diff pr.diff --report ut_cov.tar.gz --json
```

## 工具输出

### pr_utils.py 输出（在线）

| 字段 | 说明 |
|------|------|
| `pr_info` | PR 基本信息（编号、仓库、作者等） |
| `diff_content` | PR 的代码变更内容 |
| `analysis` | 变更文件分析（Pass 文件、测试文件等） |
| `ut_report` | UT-REPORT 状态（通过/失败/中止） |
| `diff_applied` | Diff 是否成功应用到本地仓库 |
| `build_status` | 编译状态 |
| `need_design_ut` | 是否需要设计 UT |

### ut_coverage.py 输出（离线）

| 字段 | 说明 |
|------|------|
| `diff_info` | Diff 变更分析结果（变更文件、代码行） |
| `coverage_info` | 覆盖率信息（总体覆盖率、文件列表等） |
| `uncovered_lines` | 未覆盖的代码行 |
| `ut_design_suggestions` | UT 设计建议列表 |

## 典型工作流

### 场景 1：为新 Pass 生成 UT

1. 分析 Pass 代码，识别关键功能
2. 在 `pypto/framework/tests/ut/passes/src/` 创建测试文件
3. 参考 `test_removeredundantop.cpp` 编写测试用例
4. 运行 `python3 build_ci.py -c -u=TestPassName.* -j=24 -f=cpp`
5. 检查覆盖率

### 场景 2：在线分析 PR

```bash
# 获取 PR diff 并分析
python3 scripts/pr_utils.py 1894

# 快速获取 UT 状态
python3 scripts/get_ut_status.py 2017
```

### 场景 3：离线分析本地文件

```bash
# 分析本地 diff 文件
python3 scripts/ut_coverage.py --diff /path/to/pr.diff

# 分析本地覆盖率报告
python3 scripts/ut_coverage.py --report /path/to/ut_cov.tar.gz

# 综合分析
python3 scripts/ut_coverage.py --diff /path/to/pr.diff --report /path/to/ut_cov.tar.gz
```

## 环境变量

| 变量 | 说明 | 必填 |
|------|------|------|
| `GITCODE_TOKEN` | GitCode API Token | 在线处理 PR 时 |
| `TILE_FWK_DEVICE_ID` | NPU 设备 ID | 运行 UT 时 |
| `PTO_TILE_LIB_CODE_PATH` | PTO 库路径 | 运行 UT 时 |

## 常见错误

### 1. API 请求失败

```
⚠️ API 请求失败: [Errno -2] Name or service not known
```

解决方案：检查网络连接，确保可以访问 gitcode.com

### 2. Diff 应用冲突

```
⚠️ Diff 应用发生冲突
```

解决方案：选择保留本地版本或接受 incoming 版本

### 3. 覆盖率文件解析失败

```
⚠️ 无法解析覆盖率文件
```

解决方案：确保文件是 .tar.gz 格式，且包含正确的 LCOV 覆盖率数据

---

## 相关文档

- [SKILL.md](../SKILL.md) - 主技能文档
- [check_list.md](./check_list.md) - 检查清单
