# pypto_pro.language.add_relu

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

先对两个tile做逐元素加法，再对结果施加ReLU激活（负值置零）。与`pypto_pro.language.add_relu_cast`的区别是不做类型转换。

## 函数原型

```python
pypto_pro.language.add_relu(out, lhs, rhs)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 目标tile，存放先加后ReLU的结果 |
| `lhs` | 输入 | 左操作数tile |
| `rhs` | 输入 | 右操作数tile |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 数据类型：FP16、FP32或INT32；该接口依次发射`TADD`与`TRELU`，类型必须同时满足两者约束<br>shape须与`lhs`、`rhs`一致；支持与`lhs`或`rhs`为同一tile，实现in-place加法+ReLU；实现会将`lhs`用作中间结果，因此`lhs`会被覆盖 |
| `lhs` | 输入 | 数据类型：与`out`一致<br>shape：与`out`一致 |
| `rhs` | 输入 | 数据类型：与`out`一致<br>shape：与`out`一致 |

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整kernel：从GM载入两个FP32输入，用`pypto_pro.language.add_relu`先加后ReLU再写回GM。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def add_relu_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], b: pl.Tensor[[64, 64], pl.DT_FP32],
                    out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.add_relu(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:add_relu:start -->
```bash
输入数据a：[[-6 -5.75 -5.5 -5.25 -5 -4.75 -4.5 -4.25 ...], [10 10.25 10.5 10.75 11 11.25 11.5 11.75 ...], [26 26.25 26.5 26.75 27 27.25 27.5 27.75 ...], [42 42.25 42.5 42.75 43 43.25 43.5 43.75 ...], ...]
输入数据b：[[1 1.5 2 2.5 3 3.5 4 4.5 ...], [33 33.5 34 34.5 35 35.5 36 36.5 ...], [65 65.5 66 66.5 67 67.5 68 68.5 ...], [97 97.5 98 98.5 99 99.5 100 100.5 ...], ...]
输出数据out：[[0 0 0 0 0 0 0 0.25 ...], [43 43.75 44.5 45.25 46 46.75 47.5 48.25 ...], [91 91.75 92.5 93.25 94 94.75 95.5 96.25 ...], [139 139.75 140.5 141.25 142 142.75 143.5 144.25 ...], ...]
```
<!-- pypto-doc-output:add_relu:end -->
