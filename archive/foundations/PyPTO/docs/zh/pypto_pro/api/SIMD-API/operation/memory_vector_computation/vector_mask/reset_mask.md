# pypto_pro.language.reset_mask

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

重置向量掩码寄存器为默认状态（全1），使后续矢量计算的所有元素均参与运算。

等价于`pypto_pro.language.set_vec_mask(-1, -1)`，即128位掩码全部置1。

## 函数原型

```python
pypto_pro.language.reset_mask()
```

无参数。

## 补充说明

向量掩码控制后续矢量计算中哪些元素参与运算。默认情况下（`section_vector()`开头），掩码为全1（所有元素活跃）。如果在计算过程中通过[`pypto_pro.language.set_vec_mask`](set_vec_mask.md)修改了掩码，需要在下一轮计算前调用`reset_mask()`恢复。

> [!CAUTION]注意
> tile级操作（如`pypto_pro.language.add`、`pypto_pro.language.mul`）内部会自动管理掩码，用户通过`pypto_pro.language.set_vec_mask`设置的掩码会被覆盖。这些API主要用于自定义操作或底层控制场景。

## 流水类型

S（标量流水）。

## 调用示例

下面是一个完整kernel：在`pypto_pro.language.add`前调用`pypto_pro.language.reset_mask()`恢复全元素掩码，验证掩码状态未损坏。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def reset_mask_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    b: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
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
        pl.reset_mask()
        pl.add(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```
