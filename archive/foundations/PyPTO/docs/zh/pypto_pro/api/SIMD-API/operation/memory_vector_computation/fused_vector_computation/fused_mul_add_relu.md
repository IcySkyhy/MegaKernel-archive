# pypto_pro.language.fused_mul_add_relu

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

融合乘加ReLU（就地）：`out = relu(out * a + b)`。先将out与a逐元素相乘，再加上b，最后对结果施加ReLU激活（负值置零）。

注意：out同时作为乘数和累加目标，运算后out的原始值被覆盖。

## 函数原型

```python
pypto_pro.language.fused_mul_add_relu(out, a, b)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输入/输出 | 目标tile，既作为乘数也存放结果 |
| `a` | 输入 | 乘数tile |
| `b` | 输入 | 加数tile |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输入/输出 | 数据类型：b16、b32<br>shape：与`a`、`b`一致 |
| `a` | 输入 | 数据类型：与`out`一致<br>shape：与`out`一致 |
| `b` | 输入 | 数据类型：与`out`一致<br>shape：与`out`一致 |

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整kernel：从GM载入三个FP16输入，用`pypto_pro.language.fused_mul_add_relu`完成`out = relu(out * a + b)`的就地融合乘加ReLU再写回GM。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl

M, N = 64, 128


@pl.jit(auto_mutex=True)
def fused_mul_add_relu_kernel(
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
        pl.fused_mul_add_relu(cur_c, cur_a, cur_b)
        pl.store(z, cur_c, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:fused_mul_add_relu:start -->
```bash
输入数据x：[[-2 -1.75 -1.5 -1.25 -1 -0.75 -0.5 -0.25 ...], [30 30.25 30.5 30.75 31 31.25 31.5 31.75 ...], [62 62.25 62.5 62.75 63 63.25 63.5 63.75 ...], [94 94.25 94.5 94.75 95 95.25 95.5 95.75 ...], ...]
输入数据y：[[1 1.125 1.25 1.375 1.5 1.625 1.75 1.875 ...], [17 17.125 17.25 17.375 17.5 17.625 17.75 17.875 ...], [33 33.125 33.25 33.375 33.5 33.625 33.75 33.875 ...], [49 49.125 49.25 49.375 49.5 49.625 49.75 49.875 ...], ...]
输入数据z原始值：[[3 2.9375 2.875 2.8125 2.75 2.6875 2.625 2.5625 ...], [-5 -5.0625 -5.125 -5.1875 -5.25 -5.3125 -5.375 -5.4375 ...], [-13 -13.0625 -13.125 -13.1875 -13.25 -13.3125 -13.375 -13.4375 ...], [-21 -21.0625 -21.125 -21.1875 -21.25 -21.3125 -21.375 -21.4375 ...], ...]
输出数据z：[[0 0 0 0 0 0 0.4375 1.234375 ...], [0 0 0 0 0 0 0 0 ...], [0 0 0 0 0 0 0 0 ...], [0 0 0 0 0 0 0 0 ...], ...]
```
<!-- pypto-doc-output:fused_mul_add_relu:end -->
