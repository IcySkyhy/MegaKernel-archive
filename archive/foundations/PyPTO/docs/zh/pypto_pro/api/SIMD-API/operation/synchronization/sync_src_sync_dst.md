# pypto_pro.language.system.sync_src / pypto_pro.language.system.sync_dst

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

AI Core内部的flag式流水同步。`sync_src`在`set_pipe`上置位flag，`sync_dst`使`wait_pipe`等待同一flag，用于约束两条具体pipe之间的执行顺序。两个接口必须成对使用。

## 函数原型

```python
pypto_pro.language.system.sync_src(*, set_pipe, wait_pipe, event_id)
pypto_pro.language.system.sync_dst(*, set_pipe, wait_pipe, event_id)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `set_pipe` | 输入 | `pypto_pro.language.PipeType`，置位flag的pipe |
| `wait_pipe` | 输入 | `pypto_pro.language.PipeType`，等待flag的pipe |
| `event_id` | 输入 | Python整数常量，或整数类型的运行时标量表达式 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `set_pipe` | 输入 | `pypto_pro.language.PipeType.MTE2`（GM→UB/L1搬入）/ `V`（向量计算）/ `MTE3`（UB→GM搬出）/ `S`（标量流水）/ `MTE1`（L1→L0搬运）/ `M`（矩阵计算）/ `FIX`（fixpipe）<br>必须是一条具体pipe，不允许`PipeType.ALL` |
| `wait_pipe` | 输入 | 取值同`set_pipe`；必须与`set_pipe`不同，否则前端报错 |
| `event_id` | 输入 | 静态ID必须是Python `int`（不接受`bool`），取值范围为`[0, 7]`<br>动态ID必须是整数类型的Scalar表达式；前端无法在编译期判定其运行时数值，用户必须保证其始终在`[0, 7]`内 |

## 配对与复用规则

- 一对`sync_src`/`sync_dst`的`set_pipe`、`wait_pipe`和`event_id`必须完全一致，且必须先置位、后等待。
- 同一event ID只能在前一次flag已被对应的`sync_dst`消费后复用。过早复用、漏写任一侧或两侧所在控制流路径不一致，都可能造成数据竞争或死锁。
- 前端只校验每次调用的参数，不会在分支、循环或函数边界上自动证明两个调用已正确配对。
- 与`auto_mutex=True`并用时，显式flag同步仍会保留；应确保它与自动mutex分别负责明确的依赖，不要为同一依赖重复同步。

## 典型同步模式

| 场景 | set_pipe | wait_pipe | 说明 |
|---|---|---|---|
| load后V才能计算 | MTE2 | V | 确保GM→UB搬运完成 |
| 计算后MTE3才能store | V | MTE3 | 确保向量计算完成 |
| store后MTE2才能load | MTE3 | MTE2 | 确保UB→GM搬出完成再搬入新数据 |

## 调用示例

下面是一个完整Kernel：从GM载入两个FP32输入，用`sync_src`/`sync_dst`约束MTE2（load）→ V（计算）→ MTE3（store）的执行顺序。该示例为纯Vector Kernel，使用`sync_src`/`sync_dst`手动同步。

```python
import pypto_pro.language as pl


@pl.jit()
def sync_src_dst_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    b: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile(tt, addr=0x0000, size=16384)
    tile_b = pl.make_tile(tt, addr=0x4000, size=16384)
    tile_out = pl.make_tile(tt, addr=0x8000, size=16384)
    with pl.section_vector():
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(tile_out, tile_a, tile_b)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tile_out, [0, 0])
```
