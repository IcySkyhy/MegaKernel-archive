---
name: precision-pruning
description: PyPTO 用例剪枝技能。在能复现精度问题的前提下，通过缩减 tile 循环次数简化测试用例。触发词：用例剪枝、剪枝、化简用例、缩减循环、pruning。
---

# 用例剪枝

> **仅做 shape 剪枝加速调试，不定位精度根因。**

缩减 tile 循环次数以加速精度调试，同时**保持原用例尾块行为不变**。

## 核心公式

对每个 tile 维度：`tail = dim % tile`

| tail | 行为 | new dim | 迭代数 |
|------|------|---------|--------|
| `= 0` | 无尾块 | `tile` | 1 轮 |
| `> 0` | 有尾块，保留 | `tile + tail` | 2 轮（1 满 + 1 尾） |

loop_unroll：`effective_tile = tile × max_unroll`，套用同上公式，tail 是元素级余数。
硬编码 `pypto.loop(N, ...)`：shape 同上，loop 字面量同步改为 1（无尾）或 2（有尾）。

## 工作流程

### 1. 分析

读代码，提取每个 tile 维度的 `dim`、`tile`、`max_unroll`：

| 循环写法 | 如何提取 |
|----------|----------|
| `m_loop = (m + tile_m - 1) // tile_m` | dim=`m`, tile=`tile_m` |
| `m_loop = (m + tile_m*4 - 1) // (tile_m*4)` + `loop_unroll` | dim=`m`, tile=`tile_m`, unroll=4 |
| `pypto.loop(2, name="B", ...)` | 硬编码，需要从测试参数中找到对应 dim/tile |

### 2. 计算

agent 读代码提取 `dim`/`tile`/`max_unroll`，传入脚本：

```python
from scripts.pruning_utils import (
    compute_pruned_dim,              # new_dim (普通 loop / loop_unroll 通用)
    compute_loop_count,              # 原始循环轮数
    compute_hardcoded_loop_count,    # 硬编码 loop 剪枝后的 N
)
```

loop_unroll 的 `max_unroll` 取 `unroll_list` 最大值。硬编码循环额外调用 `compute_hardcoded_loop_count(dim, tile)` 得新 `N`。

### 3. 修改

- **shape 驱动的循环**：只改测试入口的 shape 字面量，kernel 不动
- **硬编码循环**：改 shape + 改 `pypto.loop(N, ...)` 中的 `N`
- golden 和 tile_fwk_config 中依赖 shape 的参数同步修改
- tile size 保持不动

### 4. 验证

**必须上板跑测试**，对比剪枝前后的精度行为：

| 原用例 | 剪枝成功 ✓ | 剪枝失败 ✗ |
|--------|-----------|-----------|
| 原本正确 | 仍正确 | 出现差异 |
| 原本错误 | 仍错误 | 差异消失 → 问题在循环迭代过程中 |

失败时逐步增加迭代：`dim = tile * N + tail`，N 从 2 递增直到差异复现。

### 5. 提交

另存为 `*_pruned.py`，头部标注剪枝信息和验证状态。

## 循环模式速查

| 模式 | kernel 写法 | 剪枝要点 |
|------|------------|----------|
| tile 循环 | `m_loop = (m + tile - 1) // tile` | 改 shape，kernel 不动 |
| loop_unroll | `pypto.loop_unroll(N, unroll_list=[...])` | `compute_pruned_dim(dim, tile, max_unroll)` |
| 硬编码 | `pypto.loop(2, name="B", ...)` | 改 shape + `compute_hardcoded_loop_count(dim, tile)` |
| 非 tile | `for i in range(32)` | 跳过，无法剪枝 |

## 常见问题

**剪枝后精度差异消失**：问题在循环迭代过程中（累积误差、状态传递），不是尾块问题。按 `dim = tile * N + tail` 逐步增加迭代直到差异复现。

**buffer 不够**：检查 tile_fwk_config，按比例缩小依赖 shape 的 buffer 参数。

**多个 kernel**：分别分析，确保 shape 一致。
