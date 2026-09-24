# pypto_pro.language.sqrt

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

逐元素开平方：`out = sqrt(src)`。对源tile中每个元素求平方根，要求输入为非负值。

## 函数原型

```python
pypto_pro.language.sqrt(out, src)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 目标tile，存放逐元素开平方结果 |
| `src` | 输入 | 源tile |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 数据类型：FP16、FP32<br>shape须与`src`一致<br>支持与`src`为同一tile，实现in-place开平方 |
| `src` | 输入 | 数据类型：与`out`一致<br>shape：与`out`一致<br>元素值须非负 |

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整kernel：把FP32源tile逐元素开平方后写回GM。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def sqrt_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32],
                out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.sqrt(cur_out, cur_a)
        pl.store(out, cur_out, [0, 0])
```
