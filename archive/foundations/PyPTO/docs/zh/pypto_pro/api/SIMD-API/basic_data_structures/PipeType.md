# pypto_pro.language.PipeType

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->

## 功能说明

硬件执行单元（流水）的类型枚举，标记某个操作运行在哪条流水上。是同步控制（[`sync_src`/`sync_dst`](../operation/synchronization/sync_src_sync_dst.md)、[`mutex_lock`](../operation/synchronization/mutex_lock_mutex_unlock.md)等）里`set_pipe`/`wait_pipe`/`pipe`参数的取值来源。

昇腾芯片内部有多条并行流水，各自负责不同阶段的数据搬运和计算。正确理解每条流水的职责是写好同步控制的基础。

## 取值

| 取值 | 说明 | 硬件角色 |
|---|---|---|
| `pypto_pro.language.PipeType.MTE1` | 搬运流水1 | L1 → L0A/L0B/Bias/ScaleLeft/ScaleRight（矩阵操作数与scale搬运） |
| `pypto_pro.language.PipeType.MTE2` | 搬运流水2 | GM → L1/UB（load搬入） |
| `pypto_pro.language.PipeType.MTE3` | 搬运流水3 | UB → GM（store搬出）、UB → L1（move搬运） |
| `pypto_pro.language.PipeType.M` | 矩阵计算流水 | Cube/MAD（matmul计算） |
| `pypto_pro.language.PipeType.V` | 向量计算流水 | element-wise、reduce、cast等向量操作 |
| `pypto_pro.language.PipeType.S` | 标量流水 | getval/setval等标量操作 |
| `pypto_pro.language.PipeType.FIX` | fixpipe流水 | 累加器结果读出（L0C → UB/GM/L1）、quantization/反量化等 |
| `pypto_pro.language.PipeType.ALL` | 本AI Core的全部流水 | V/M/MTE1/MTE2/MTE3/FIX等流水 |

## 补充说明

[`load`](../operation/memory_data_movement/load.md)、[`move`](../operation/memory_data_movement/move.md)和[`store`](../operation/memory_data_movement/store.md)的流水由源/目的内存空间自动决定：

| 操作 | 源 → 目的 | 流水 |
|---|---|---|
| `load` | GM → L1/UB | MTE2 |
| `store` | Vec(UB) → GM | MTE3 |
| `store` | Acc(L0C) → GM | FIX |
| `move` | Mat(L1) → Left/Right(L0A/L0B)/Bias/ScaleLeft/ScaleRight | MTE1 |
| `move` | Mat(L1) → Scaling | FIX |
| `move` | Mat(L1) → Vec(UB) | V |
| `move` | Acc(L0C) → Vec(UB) | FIX |
| `move` | Vec(UB) → Vec(UB) | V |
| `move` | Vec(UB) → Mat(L1) | MTE3 |
| `move` | 其余 | V |

典型同步模式：

| 场景 | set_pipe | wait_pipe | 说明 |
|---|---|---|---|
| load后V才能计算 | MTE2 | V | 确保GM→UB搬运完成 |
| 计算后MTE3才能store | V | MTE3 | 确保向量计算完成 |
| store后MTE2才能load | MTE3 | MTE2 | 确保UB→GM搬出完成再搬入新数据 |
| L1→L0A搬运后M才能计算 | MTE1 | M | 确保矩阵操作数就位 |
| matmul后FIX才能读出结果 | M | FIX | 确保矩阵计算完成 |
| 标量读写 | MTE2/MTE3 | S | getval/setval走标量流水 |

## 调用示例

```python
import pypto_pro.language as pl
# load 后同步：MTE2 置位，V 等待
pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

# 计算后同步：V 置位，MTE3 等待
pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
```
