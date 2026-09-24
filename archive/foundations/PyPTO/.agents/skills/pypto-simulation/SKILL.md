---
name: pypto-simulation
description: PyPTO Soc CAModel 仿真执行技能。用于在 CPU 上模拟昇腾 AI 处理器行为，生成核内流水报告。触发词：Soc CAModel 仿真、核内流水、流水报告。
---

# PyPTO Soc CAModel 仿真执行

## 概述

Soc CAModel 仿真在 CPU 上模拟昇腾 AI 处理器，无需 NPU 硬件即可分析核内流水。

**关键特点：**
- 仿真速度比真实 NPU 慢 100-1000 倍
- 需充足资源：CPU >16核，内存 >32GB
- 生成 Chrome Tracing 流水图

---

## 核心流程（3 步）

### 步骤 1：检查并安装 PyPTO【必须】

> **执行目录：pypto 仓库主目录**

```bash
cd /path/to/pypto
pip show pypto
```

**检查点：**
- ✅ 输出包含 `Name: pypto` 和 `Version: x.x.x` → PyPTO 已安装，跳过安装
- ❌ 输出为空或报错 → PyPTO 未安装，需执行安装

**如 PyPTO 未安装，在主目录下执行：**

```bash
cd /path/to/pypto
# 编译产出 run
python3 build_ci.py --clean --no_isolation
# 安装到当前环境
bash build_out/cann-pypto_*.run --full -q --pylocal
```

安装完成后重新检查：
```bash
pip show pypto
```

---

### 步骤 2：执行仿真【必须】

> **执行目录：pypto 仓库主目录**

**前置检查（所有模式执行前必须完成）：**

读取用户示例脚本，确认并修改：

- **tensor 在 CPU 分配**：脚本中所有 torch tensor（`torch.rand`/`torch.empty`/`torch.ones`/`torch.zeros` 等）的 `device` 参数须为 `'cpu'` 或省略；若为 `'npu:x'` 需改为 `'cpu'`

根据用户目的选择模式：

**模式 A：任务级仿真（粒度较粗、速度快但性能仿真准确度较低）**

```bash
cd /path/to/pypto
python3 examples/00_hello_world/hello_world.py --run_mode sim
```

**模式 B：指令级仿真（粒度较细、性能仿真准确度高但速度慢，查看流水报告）**

**额外前置检查：**

- **设置 accuracy_level=2**：在脚本中 jit 调用前添加 `pypto.set_global_config("simulation.accuracy_level", 2)`，用于开启核内流水采集；若缺失会导致无法生成流水报告

```bash
cd /path/to/pypto
cannsim record 'python3 examples/00_hello_world/hello_world.py --run_mode sim' -s Ascend950 -n 0 -g -o output/
```

> ⚠️ **核数说明**：上述命令带 `-n 0`，执行后**默认只能查看 0 核**的流水报告。若需查看多核流水，执行时**不要指定 `-n 0`**（去掉该参数），仿真器会对所有核开启日志采集；随后通过步骤 3 的 `--core-id all`（或指定核号如 `--core-id 1,5`）生成对应核的报告。

**检查点：**
- 用例输出包含 `passed` 或精度校验通过（如 `Max difference` 在容差内）
- 主目录下生成 `output/cannsim_*/` 目录
- 模式 B 额外：`output/cannsim_*/report/` 下生成 `trace_core0.json`

**模式 B 执行完成后，告知用户可选步骤 3（模式 A 不需要）：**
>
> **可选步骤 3（仅模式 B）：** 已生成单核报告；如需全部核（32核）的完整流水报告，请确认后执行：
> ```bash
> cannsim report -e <cannsim_dir> -o <cannsim_dir>/report --core-id all
> ```

---
**查看报告：**
- Chrome 浏览器 → `chrome://tracing` → Load `trace_core*.json`

---

## 实际案例：hello_world

> **所有命令均在 pypto 仓库主目录执行**

```bash
# 步骤1：检查PyPTO安装（必须）
$ cd /path/to/pypto
$ pip show pypto
Name: pypto
Version: 0.2.1

# 如果未安装：
$ python3 build_ci.py --clean --no_isolation
$ bash build_out/cann-pypto_*.run --full -q --pylocal

# 步骤2：执行仿真（必须）
# 模式A：任务级仿真
$ cd /path/to/pypto
$ python3 examples/00_hello_world/hello_world.py --run_mode sim

# 模式B：指令级仿真（脚本中需设置 pypto.set_global_config("simulation.accuracy_level", 2)）
$ cd /path/to/pypto
$ cannsim record 'python3 examples/00_hello_world/hello_world.py --run_mode sim' -s Ascend950 -n 0 -g -o output/
Json Saved at: .../report/trace_core0.json ✓

# 步骤3：生成全部核报告（可选，需用户确认）
$ cd /path/to/pypto
$ cannsim report -e <cannsim_dir> -o <cannsim_dir>/report --core-id all
Json Saved: trace_core0.json ... trace_core31.json ✓
```

### 输出结果

```
output/cannsim_*/
├── cannsim.log              # 仿真日志
└── record/
    └── instr.bin            # 指令二进制（cannsim record 采集）

# 模式 B（带 -g）自动生成报告：
output/cannsim_*/
└── report/
    ├── results/
    └── trace_reports/
        └── trace_core0.json # 单核流水图

# 步骤3执行后追加：
output/cannsim_*/report/trace_reports/
├── trace_core0.json
├── trace_core1.json
└── ... (trace_core31.json，共32个核)
```

---

## 关键注意事项

### 步骤执行顺序

- **必须严格按 1→2→3 顺序执行**
- **所有步骤均在 pypto 仓库主目录下执行**
- 步骤 3 为可选（仅模式 B 适用，模式 A 不需要），其他步骤均为必须
- 每个步骤完成后需检查对应的检查点

### 可选步骤执行规范

步骤 3（生成全部核报告，仅模式 B 适用）需遵循以下规范：
- **不得主动执行**，需等待用户确认
- 模式 B 步骤 2 执行完成后，告知用户已生成单核报告，并说明可选步骤 3
- 用户明确要求时，方可执行 `cannsim report --core-id all`

---

## 参考文档

| 文档 | 内容 | 加载时机 |
|------|------|----------|
| [references/troubleshooting.md](references/troubleshooting.md) | 仿真失败诊断与修复（checkSqeSupported 断言失败、超时、段错误等） | 遇到仿真失败时按需读取 |

> 遇到仿真失败时，读取 [references/troubleshooting.md](references/troubleshooting.md) 按错误特征定位章节。

---

## 命令速查

```bash
# 任务级仿真
python3 examples/00_hello_world/hello_world.py --run_mode sim

# 指令级仿真（脚本中需设置 pypto.set_global_config("simulation.accuracy_level", 2)）
cannsim record 'python3 examples/00_hello_world/hello_world.py --run_mode sim' -s Ascend950 -n 0 -g -o output/

# 生成完整报告（全部核）
cannsim report -e <cannsim_dir> -o <cannsim_dir>/report --core-id all

# 查看输出目录
ls output/cannsim_*/

# 查看报告
ls output/cannsim_*/report/
```

---

## 触发词

- Soc CAModel 仿真、核内流水、流水报告、任务级仿真、指令级仿真

---

## 相关技能

- `pypto-environment-setup`：环境准备
- `tune-incore`：核内流水优化（基于 Soc CAModel 报告）
- `tune-swimlane`：泳道图分析优化
