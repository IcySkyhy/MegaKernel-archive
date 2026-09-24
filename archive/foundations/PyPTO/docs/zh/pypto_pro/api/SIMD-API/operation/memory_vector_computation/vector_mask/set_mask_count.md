# pypto_pro.language.set_mask_count

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

将向量掩码寄存器切换为**counter（计数）模式**。在该模式下，掩码寄存器的低64位被解释为“参与运算的元素个数”，而非逐位掩码。

与[`pypto_pro.language.set_vec_mask`](set_vec_mask.md)配合使用：先切换到count模式，再设置元素个数，即可精确控制尾块计算的有效元素数。

## 函数原型

```python
pypto_pro.language.set_mask_count()
```

无参数。

## 补充说明

count模式下，[`pypto_pro.language.set_vec_mask`](set_vec_mask.md)的`mask_low`参数被解释为元素个数（而非位掩码）。用于处理尾块（最后一块tile的有效列数小于tile列数）场景。

> [!CAUTION]注意
> tile级操作（如`pypto_pro.language.add`、`pypto_pro.language.mul`）内部会自动管理掩码，用户通过`pypto_pro.language.set_vec_mask`设置的掩码会被覆盖。这些API主要用于自定义操作或底层控制场景。

## 流水类型

S（标量流水）。

## 调用示例

下面是一个完整kernel：演示`set_mask_count` + `set_vec_mask` + `set_mask_norm`的模式切换流程，最终恢复norm模式后做`pypto_pro.language.add`。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def mask_count_norm_kernel(
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
        pl.set_mask_count()
        pl.set_vec_mask(0, 64 * 64)
        pl.set_mask_norm()
        pl.set_vec_mask(-1, -1)
        pl.add(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```
