# pypto_pro.language.partmin

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

对两路输入的有效区域执行部分逐元素最小值。两个输入在同一位置均有效时写入二者最小值；只有一路有效时复制该路输入。

## 函数原型

```python
pypto_pro.language.partmin(out, src0, src1)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 目标Tile |
| `src0` | 输入 | 第一条源Tile |
| `src1` | 输入 | 第二条源Tile |

## 参数范围

三个参数均须为Vec Tile，dtype相同，支持INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、FP16、BF16和FP32。两路输入中至少一路的有效shape须与`out`一致；另一条输入的有效行数和列数不得超过`out`。`out`的有效区域为零时操作直接返回。

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整kernel：`ty`仅前32行有效，用`partmin`取两路输入的有效区域最小值，后32行保持`tx`。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def partmin_kernel(x: pl.Tensor[[64, 128], pl.DT_FP16],
                   y: pl.Tensor[[64, 128], pl.DT_FP16],
                   out: pl.Tensor[[64, 128], pl.DT_FP16]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16,
                     target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    gx = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    gy = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    gout = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        tx, ty, tout = gx.current(), gy.current(), gout.current()
        pl.set_validshape(tx, [64, 128])
        pl.load(tx, x, [0, 0])
        pl.set_validshape(ty, [32, 128])
        pl.load(ty, y, [0, 0])
        pl.set_validshape(tout, [64, 128])
        pl.partmin(tout, tx, ty)
        pl.store(out, tout, [0, 0])
```

前32行取`minimum(x, y)`，后32行只有`x`有效，因此复制`x`。
