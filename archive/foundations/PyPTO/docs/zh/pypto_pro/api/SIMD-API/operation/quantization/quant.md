# pypto_pro.language.quant

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

把高精度tile量化为低精度整型（支持对称/非对称）。对称模式：`out = clamp(round(src * scale), -128, 127)`；非对称模式：`out = clamp(round(src * scale) + offset, 0, 255)`。

## 函数原型

```python
pypto_pro.language.quant(out, src, scale, *, mode=pl.QuantMode.SYM, offset=None)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 量化结果tile（低精度整型） |
| `src` | 输入 | 源tile（高精度浮点） |
| `scale` | 输入 | 量化缩放系数tile |
| `mode` | 输入 | 量化模式，默认`pl.QuantMode.SYM` |
| `offset` | 输入 | 非对称模式下的零点偏移tile |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 数据类型：`pypto_pro.language.DT_INT8`（对称模式，范围[-128, 127]）或`pypto_pro.language.DT_UINT8`（非对称模式，范围[0, 255]）<br>shape须与`src`一致 |
| `src` | 输入 | 数据类型：`pypto_pro.language.DT_FP32`<br>shape：与`out`一致 |
| `scale` | 输入 | 数据类型：`pypto_pro.language.DT_FP32`<br>按行提供量化系数，典型shape为`[src行数, 1]` |
| `mode` | 输入 | `pl.QuantMode.SYM`（对称，默认）或`pl.QuantMode.ASYM`（非对称）<br>非对称模式时`offset`必填 |
| `offset` | 输入 | 数据类型：`pypto_pro.language.DT_FP32`<br>shape：与`scale`一致<br>仅`pl.QuantMode.ASYM`模式需要，`pl.QuantMode.SYM`模式忽略 |

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整kernel：从GM载入FP32源tile和per-row scale，用`pypto_pro.language.quant`对称量化为INT8再写回GM。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def quant_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP32],
    scale: pl.Tensor[[64, 1], pl.DT_FP32],
    out: pl.Tensor[[64, 128], pl.DT_INT8],
):
    tile_src = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                                  addrs=0x0000, mutex_ids=[0])
    tile_scale = pl.make_tile_group(type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                                    addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=pl.TileType(shape=[64, 128], dtype=pl.DT_INT8, target_memory=pl.MemorySpace.Vec),
                                  addrs=0xA000, mutex_ids=[2])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_scale = tile_scale.current()
        cur_out = tile_out.current()
        pl.load(cur_src, src, [0, 0])
        pl.load(cur_scale, scale, [0, 0])
        pl.quant(cur_out, cur_src, cur_scale, mode=pl.QuantMode.SYM)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:quant:start -->
```bash
输入数据src：[[-16 -15.75 -15.5 -15.25 -15 -14.75 -14.5 -14.25 ...], [16 16.25 16.5 16.75 17 17.25 17.5 17.75 ...], [48 48.25 48.5 48.75 49 49.25 49.5 49.75 ...], [80 80.25 80.5 80.75 81 81.25 81.5 81.75 ...], ...]
输入数据scale：[[4], [4], [4], [4], ...]
输出数据out：[[-64 -63 -62 -61 -60 -59 -58 -57 ...], [64 65 66 67 68 69 70 71 ...], [127 127 127 127 127 127 127 127 ...], [127 127 127 127 127 127 127 127 ...], ...]
```
<!-- pypto-doc-output:quant:end -->
