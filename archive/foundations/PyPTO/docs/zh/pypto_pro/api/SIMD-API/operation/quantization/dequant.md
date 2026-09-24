# pypto_pro.language.dequant

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

把低精度整型tile反量化回高精度浮点。计算公式：`out = (src - offset) * scale`。

## 函数原型

```python
pypto_pro.language.dequant(out, src, scale, offset)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 反量化结果tile（高精度浮点） |
| `src` | 输入 | 源tile（低精度整型） |
| `scale` | 输入 | 缩放系数tile |
| `offset` | 输入 | 零点偏移tile |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 数据类型：`pypto_pro.language.DT_FP32`<br>shape须与`src`一致 |
| `src` | 输入 | 数据类型：`pypto_pro.language.DT_INT8`或`pypto_pro.language.DT_INT16`<br>shape和valid shape须与`out`一致，且采用行主序布局 |
| `scale` | 输入 | 数据类型：`pypto_pro.language.DT_FP32`<br>按行提供参数，有效行数须等于`out`的有效行数；典型shape为`[行数, 1]` |
| `offset` | 输入 | 数据类型：`pypto_pro.language.DT_FP32`<br>shape：与`scale`一致，有效行数须等于`out`的有效行数<br>对称量化场景可传全零tile |

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整kernel：从GM载入INT8源tile、per-row scale和offset，用`pypto_pro.language.dequant`反量化为FP32再写回GM。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def dequant_kernel(
    src: pl.Tensor[[64, 128], pl.DT_INT8],
    scale: pl.Tensor[[64, 1], pl.DT_FP32],
    offset: pl.Tensor[[64, 1], pl.DT_FP32],
    out: pl.Tensor[[64, 128], pl.DT_FP32],
):
    tile_src = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec),
                                  addrs=0x0000, mutex_ids=[0])
    tile_scale = pl.make_tile_group(type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                                    addrs=0x4000, mutex_ids=[1])
    tile_offset = pl.make_tile_group(type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                                     addrs=0x5000, mutex_ids=[2])
    tile_out = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                                  addrs=0x6000, mutex_ids=[3])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_scale = tile_scale.current()
        cur_offset = tile_offset.current()
        cur_out = tile_out.current()
        pl.load(cur_src, src, [0, 0])
        pl.load(cur_scale, scale, [0, 0])
        pl.load(cur_offset, offset, [0, 0])
        pl.dequant(cur_out, cur_src, cur_scale, cur_offset)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:dequant:start -->
```bash
输入数据src：[[-32 -31 -30 -29 -28 -27 -26 -25 ...], [-32 -31 -30 -29 -28 -27 -26 -25 ...], [-32 -31 -30 -29 -28 -27 -26 -25 ...], [-32 -31 -30 -29 -28 -27 -26 -25 ...], ...]
输入数据scale：[[0.25], [0.25], [0.25], [0.25], ...]
输入数据offset：[[0], [0], [0], [0], ...]
输出数据out：[[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], ...]
```
<!-- pypto-doc-output:dequant:end -->
