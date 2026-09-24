# pypto_pro.language.mul_add_dst

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

融合乘加到目标：`out = lhs * rhs + out`。先计算lhs和rhs的逐元素乘积，再累加到out的现有值上。

## 函数原型

```python
pypto_pro.language.mul_add_dst(out, lhs, rhs)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输入/输出 | 目标tile，既提供累加初值也存放结果 |
| `lhs` | 输入 | 左操作数tile |
| `rhs` | 输入 | 右操作数tile |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输入/输出 | 数据类型：INT16、UINT16、INT32、UINT32、INT64、UINT64、FP16、BF16或FP32；该接口依次发射`TMUL`与`TADD`，类型必须同时满足两者约束<br>shape：与`lhs`、`rhs`一致 |
| `lhs` | 输入 | 数据类型：与`out`一致<br>shape：与`out`一致 |
| `rhs` | 输入 | 数据类型：与`out`一致<br>shape：与`out`一致 |

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整kernel：从GM载入三个FP16输入，用`pypto_pro.language.mul_add_dst`完成`out = lhs * rhs + out`的就地融合乘加再写回GM。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl

M, N = 64, 128


@pl.jit(auto_mutex=True)
def mul_add_dst_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    y: pl.Tensor[[M, N], pl.DT_FP16],
    z: pl.Tensor[[M, N], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    tile_c = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_c = tile_c.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_b, y, [0, 0])
        pl.load(cur_c, z, [0, 0])
        pl.mul_add_dst(cur_c, cur_a, cur_b)
        pl.store(z, cur_c, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:mul_add_dst:start -->
```bash
输入数据x：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [65 65.25 65.5 65.75 66 66.25 66.5 66.75 ...], [97 97.25 97.5 97.75 98 98.25 98.5 98.75 ...], ...]
输入数据y：[[3 2.875 2.75 2.625 2.5 2.375 2.25 2.125 ...], [-13 -13.125 -13.25 -13.375 -13.5 -13.625 -13.75 -13.875 ...], [-29 -29.125 -29.25 -29.375 -29.5 -29.625 -29.75 -29.875 ...], [-45 -45.125 -45.25 -45.375 -45.5 -45.625 -45.75 -45.875 ...], ...]
输入数据z原始值：[[-2 -1.9375 -1.875 -1.8125 -1.75 -1.6875 -1.625 -1.5625 ...], [6 6.0625 6.125 6.1875 6.25 6.3125 6.375 6.4375 ...], [14 14.0625 14.125 14.1875 14.25 14.3125 14.375 14.4375 ...], [22 22.0625 22.125 22.1875 22.25 22.3125 22.375 22.4375 ...], ...]
输出数据z：[[1 1.65625 2.25 2.78125 3.25 3.65625 4 4.28125 ...], [-423 -430.5 -438 -445.25 -452.75 -460.5 -468 -475.75 ...], [-1.871000e+03 -1.886000e+03 -1.902000e+03 -1.917000e+03 -1.933000e+03 -1.949000e+03 -1.964000e+03 -1.980000e+03 ...], [-4.344000e+03 -4.364000e+03 -4.388000e+03 -4.412000e+03 -4.436000e+03 -4.460000e+03 -4.484000e+03 -4.508000e+03 ...], ...]
```
<!-- pypto-doc-output:mul_add_dst:end -->
