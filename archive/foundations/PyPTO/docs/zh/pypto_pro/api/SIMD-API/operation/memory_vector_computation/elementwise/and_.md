# pypto_pro.language.and_

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

两个操作数对应位置逐元素按位与。支持tile-tile和tile-scalar（scalar指标量）两种模式，支持in-place写法（`out`与`lhs`为同一tile）。

- **tile-tile模式**：`and_(out, lhs, rhs)` -> `out[i] = lhs[i] & rhs[i]`
- **tile-scalar模式**：`and_(out, lhs, scalar)` -> `out[i] = lhs[i] & scalar`

## 函数原型

```python
pypto_pro.language.and_(out, lhs, rhs)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 目标tile，存放逐元素按位与结果 |
| `lhs` | 输入 | 左操作数tile |
| `rhs` | 输入 | 右操作数（tile或scalar） |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 数据类型：8/16/32位整型<br>shape须与`lhs`一致；支持与`lhs`为同一tile，实现in-place按位与 |
| `lhs` | 输入 | 数据类型：与`out`一致<br>shape：与`out`一致 |
| `rhs` | 输入 | tile-tile模式：数据类型与`out`一致，shape与`out`一致<br>tile-scalar模式：scalar值（int/Scalar） |

## 流水类型

V（向量计算流水）。

## 调用示例

### tile-scalar模式

下面是一个完整kernel：从GM载入INT32输入到UB，用`pypto_pro.language.and_`与标量`7`逐元素按位与后写回GM。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def and_scalar_kernel(a: pl.Tensor[[64, 64], pl.DT_INT32],
                      out: pl.Tensor[[64, 64], pl.DT_INT32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.and_(cur_out, cur_a, 7)
        pl.store(out, cur_out, [0, 0])
```

### tile-tile模式

```python
# 两个 tile 对应位置按位与
pl.and_(tile_out, tile_a, tile_b)
```
