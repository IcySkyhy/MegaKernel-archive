# pypto_pro.language.cast

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

把tile中的元素转成另一种数据类型，支持多种舍入模式来控制精度损失。

## 函数原型

```python
pypto_pro.language.cast(out, src, *, mode=pl.RoundMode.CAST_ROUND)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 目标tile，存放类型转换后的结果 |
| `src` | 输入 | 源tile |
| `mode` | 输入 | 舍入模式，默认`pl.RoundMode.CAST_ROUND` |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | Vec（UB）Tile，目标数据类型由`out.dtype`决定；物理shape和`valid_shape`须与`src`一致 |
| `src` | 输入 | Vec（UB）Tile；物理shape和`valid_shape`须与`out`一致。支持的源/目标dtype组合由`TCVT`实现决定，包括常用8/16/32/64位整数、FP16、BF16、FP32及部分FP4/FP8/HF8转换；并非任意同位宽或跨位宽组合都合法 |
| `mode` | 输入 | 舍入模式：`pl.RoundMode.CAST_NONE` / `pl.RoundMode.CAST_RINT` / `pl.RoundMode.CAST_ROUND` / `pl.RoundMode.CAST_FLOOR` / `pl.RoundMode.CAST_CEIL` / `pl.RoundMode.CAST_TRUNC` / `pl.RoundMode.CAST_ODD`。具体转换必须支持所选模式；`CAST_ODD`当前仅对FP32→FP16实现奇数舍入，其他组合不应使用该模式 |

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整kernel：把FP16源tile转为FP32写回GM。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def cast_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP16],
    out: pl.Tensor[[64, 128], pl.DT_FP32],
):
    tt_in = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tt_out = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_src = pl.make_tile_group(type=tt_in, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt_out, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_out = tile_out.current()
        pl.load(cur_src, src, [0, 0])
        pl.cast(cur_out, cur_src, mode=pl.RoundMode.CAST_ROUND)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:cast:start -->
```bash
输入数据src：[[-4 -3.75 -3.5 -3.25 -3 -2.75 -2.5 -2.25 ...], [28 28.25 28.5 28.75 29 29.25 29.5 29.75 ...], [60 60.25 60.5 60.75 61 61.25 61.5 61.75 ...], [92 92.25 92.5 92.75 93 93.25 93.5 93.75 ...], ...]
输出数据out：[[-4 -3.75 -3.5 -3.25 -3 -2.75 -2.5 -2.25 ...], [28 28.25 28.5 28.75 29 29.25 29.5 29.75 ...], [60 60.25 60.5 60.75 61 61.25 61.5 61.75 ...], [92 92.25 92.5 92.75 93 93.25 93.5 93.75 ...], ...]
```
<!-- pypto-doc-output:cast:end -->

其他典型用法（节选）：

```python
# FP32 → FP16（缩窄）
pl.cast(p_f16, qk_vec, mode=pl.RoundMode.CAST_ROUND)

# FP16 → FP32（扩展，无舍入）
pl.cast(fp32_tile, key_row_tile, mode=pl.RoundMode.CAST_NONE)
```
