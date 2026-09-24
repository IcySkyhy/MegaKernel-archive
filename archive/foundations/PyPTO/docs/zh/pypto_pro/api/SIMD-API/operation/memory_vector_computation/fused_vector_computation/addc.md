# pypto_pro.language.addc

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

三个tile逐元素相加：`out = a + b + c`。将三个操作数对应位置的元素相加，结果写入`out`。

## 函数原型

```python
pypto_pro.language.addc(out, a, b, c)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 目标tile，存放逐元素三数相加结果 |
| `a` | 输入 | 第一个输入tile |
| `b` | 输入 | 第二个输入tile |
| `c` | 输入 | 第三个输入tile |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 数据类型：b16、b32<br>shape：与`a`、`b`、`c`一致 |
| `a` | 输入 | 数据类型：与`out`一致<br>shape：与`out`一致 |
| `b` | 输入 | 数据类型：与`out`一致<br>shape：与`out`一致 |
| `c` | 输入 | 数据类型：与`out`一致<br>shape：与`out`一致 |

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整kernel：从GM载入三个FP32输入，用`pypto_pro.language.addc`完成`out = a + b + c`的三数融合加法再写回GM。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def addc_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32],
                b: pl.Tensor[[64, 64], pl.DT_FP32],
                c: pl.Tensor[[64, 64], pl.DT_FP32],
                out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_c = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    tile_out = pl.make_tile_group(type=tt, addrs=0xC000, mutex_ids=[3])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_c = tile_c.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.load(cur_c, c, [0, 0])
        pl.addc(cur_out, cur_a, cur_b, cur_c)
        pl.store(out, cur_out, [0, 0])
```
