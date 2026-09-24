# pypto_pro.language.system.mutex_lock / pypto_pro.language.system.mutex_unlock

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

基于buffer ID的AI Core内部流水互斥接口。`mutex_lock`在指定pipe上获取`mutex_id`对应的缓冲区互斥资源，`mutex_unlock`释放该资源。它们用于防止多条pipe在数据尚未生产完或消费完时复用同一片上缓冲区。

## 函数原型

```python
pypto_pro.language.system.mutex_lock(*, pipe, mutex_id, max_mutex_id=2, mutex_ids=None)
pypto_pro.language.system.mutex_unlock(*, pipe, mutex_id, max_mutex_id=2, mutex_ids=None)
```

## 参数类型与范围

| 参数 | 输入/输出 | 类型、范围与语义 |
|---|---|---|
| `pipe` | 输入 | `pypto_pro.language.PipeType`，必须是`MTE1`/`MTE2`/`MTE3`/`V`/`M`/`S`/`FIX`中的一条具体pipe；不允许`PipeType.ALL` |
| `mutex_id` | 输入 | Python整数常量，或整数类型的运行时Scalar表达式。静态ID的取值范围为`[0, 31]`，不接受`bool`；动态ID的运行时数值也必须在`[0, 31]`内 |
| `max_mutex_id` | 输入 | 编译期mutex分析使用的候选数量，默认为`2`。必须是`[1, 32]`内的Python `int`，不接受`bool` |
| `mutex_ids` | 输入 | 编译期mutex分析使用的可选候选ID列表。可为`None`或非空`list`/`tuple`；每个元素必须是`[0, 31]`内的Python `int`，不接受`bool` |

如显式传入`max_mutex_id`或`mutex_ids`，前端都会执行相应的类型和范围校验。生成代码直接使用`mutex_id`；当它是运行时表达式时，用户必须保证其实际取值在`[0, 31]`内。

## 配对与使用规则

- `mutex_lock`与`mutex_unlock`必须对同一`pipe`和同一`mutex_id`成对使用，且必须先加锁、后解锁。
- 漏解锁、重复获取同一pipe上未释放的ID，或者使加锁和解锁处于不对称的控制流路径，都可能导致死锁或缓冲区竞争。
- 前端只校验每次调用的参数，不会在分支、循环或函数边界上自动证明lock/unlock已正确配对。
- `auto_mutex=True`仅对带mutex元数据的Tile自动生成互斥操作；显式的`mutex_lock`/`mutex_unlock`仍会保留。自动和手动管理可以在同一Kernel中并存，但不应对同一pipe/ID的同一次访问重复加锁。

常规单缓冲、双缓冲和N缓冲场景推荐使用[`make_tile_group`](../resource_management/make_tile_group.md)配合`auto_mutex=True`。需要精确控制加锁pipe和插入位置时，再使用本页手动接口。

## 调用示例

下面的Kernel在`auto_mutex=False`时计算`out = x + x`。输入UB使用mutex ID 0约束MTE2和V的访问顺序，输出UB使用mutex ID 1约束V和MTE3的访问顺序。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=False)
def mutex_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_x = pl.make_tile(tt, addr=0x0000, size=16384)
    tile_out = pl.make_tile(tt, addr=0x4000, size=16384)
    with pl.section_vector():
        pl.system.mutex_lock(pipe=pl.PipeType.MTE2, mutex_id=0)
        pl.load(tile_x, x, [0, 0])
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE2, mutex_id=0)

        pl.system.mutex_lock(pipe=pl.PipeType.V, mutex_id=0)
        pl.system.mutex_lock(pipe=pl.PipeType.V, mutex_id=1)
        pl.add(tile_out, tile_x, tile_x)
        pl.system.mutex_unlock(pipe=pl.PipeType.V, mutex_id=1)
        pl.system.mutex_unlock(pipe=pl.PipeType.V, mutex_id=0)

        pl.system.mutex_lock(pipe=pl.PipeType.MTE3, mutex_id=1)
        pl.store(out, tile_out, [0, 0])
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE3, mutex_id=1)
```
