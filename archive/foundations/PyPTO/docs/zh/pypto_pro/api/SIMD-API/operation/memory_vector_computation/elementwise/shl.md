# pypto_pro.language.shl

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

逐元素左移：`out = lhs << rhs`。`rhs`为逐元素移位量Tile。

## 函数原型

```python
pypto_pro.language.shl(out, lhs, rhs)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 左移结果Tile |
| `lhs` | 输入 | 被移位Tile |
| `rhs` | 输入 | 移位量Tile |

## 参数范围

`out`、`lhs`和`rhs`须为Vec、row-major Tile，数据类型均为INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64或UINT64，且类型和有效shape一致。`rhs`中的移位量应非负并小于元素位宽。

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整kernel：从GM载入两个INT32输入到UB，用`pypto_pro.language.shl`逐元素左移后写回GM。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def shl_kernel(a: pl.Tensor[[64, 64], pl.DT_INT32],
               b: pl.Tensor[[64, 64], pl.DT_INT32],
               out: pl.Tensor[[64, 64], pl.DT_INT32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.shl(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```
