# pypto_pro.language.sub_relu

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

先对两个tile做逐元素减法，再对结果施加ReLU激活（负值置零）。与`pypto_pro.language.sub_relu_cast`的区别是不做类型转换。

## 函数原型

```python
pypto_pro.language.sub_relu(out, lhs, rhs)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 目标tile，存放先减后ReLU的结果 |
| `lhs` | 输入 | 左操作数tile（被减数） |
| `rhs` | 输入 | 右操作数tile（减数） |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 数据类型：FP16或FP32；该接口依次发射`TSUB`与`TRELU`，类型必须同时满足两者约束<br>shape须与`lhs`、`rhs`一致；支持与`lhs`或`rhs`为同一tile，实现in-place减法+ReLU；实现会将`lhs`用作中间结果，因此`lhs`会被覆盖 |
| `lhs` | 输入 | 数据类型：与`out`一致<br>shape：与`out`一致 |
| `rhs` | 输入 | 数据类型：与`out`一致<br>shape：与`out`一致 |

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整kernel：从GM载入两个FP32输入，用`pypto_pro.language.sub_relu`先减后ReLU再写回GM。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def sub_relu_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], b: pl.Tensor[[64, 64], pl.DT_FP32],
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
        pl.sub_relu(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:sub_relu:start -->
```bash
输入数据a：[[2 2.5 3 3.5 4 4.5 5 5.5 ...], [34 34.5 35 35.5 36 36.5 37 37.5 ...], [66 66.5 67 67.5 68 68.5 69 69.5 ...], [98 98.5 99 99.5 100 100.5 101 101.5 ...], ...]
输入数据b：[[3 3.25 3.5 3.75 4 4.25 4.5 4.75 ...], [19 19.25 19.5 19.75 20 20.25 20.5 20.75 ...], [35 35.25 35.5 35.75 36 36.25 36.5 36.75 ...], [51 51.25 51.5 51.75 52 52.25 52.5 52.75 ...], ...]
输出数据out：[[0 0 0 0 0 0.25 0.5 0.75 ...], [15 15.25 15.5 15.75 16 16.25 16.5 16.75 ...], [31 31.25 31.5 31.75 32 32.25 32.5 32.75 ...], [47 47.25 47.5 47.75 48 48.25 48.5 48.75 ...], ...]
```
<!-- pypto-doc-output:sub_relu:end -->
