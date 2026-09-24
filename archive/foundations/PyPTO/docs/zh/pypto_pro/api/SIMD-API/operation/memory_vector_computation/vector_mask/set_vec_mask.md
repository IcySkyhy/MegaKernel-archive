# pypto_pro.language.set_vec_mask

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

显式设置向量掩码寄存器的128位值（高64位 + 低64位），按位精确指定哪些元素参与后续矢量计算。

掩码寄存器的含义取决于当前模式：

- **norm模式**（默认）：128位逐位掩码，每一位对应一个元素是否活跃
- **count模式**：`mask_low`被解释为有效元素个数，`mask_high`忽略

模式切换通过[`pypto_pro.language.set_mask_norm`](set_mask_norm.md) / [`pypto_pro.language.set_mask_count`](set_mask_count.md)控制。

## 函数原型

```python
pypto_pro.language.set_vec_mask(mask_high, mask_low)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `mask_high` | 输入 | 掩码高64位 |
| `mask_low` | 输入 | 掩码低64位 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `mask_high` | 输入 | 表示64位掩码位模式的整型常量或运行时整型标量表达式<br>norm模式下控制第64-127号元素的活跃状态<br>count模式下忽略 |
| `mask_low` | 输入 | 表示64位掩码位模式的整型常量或运行时整型标量表达式<br>norm模式下控制第0-63号元素的活跃状态<br>count模式下为有效元素个数 |

## 补充说明

常用掩码值：

| 值 | 含义 |
|---|---|
| `(-1, -1)` | 全1，所有元素活跃（等价于[`pypto_pro.language.reset_mask()`](reset_mask.md)） |
| `(0, 0)` | 全0，所有元素不活跃 |
| `(0, 0xFFFFFFFF)` | 低32个元素活跃 |
| `(0, valid_cols)` | count模式下前`valid_cols`个元素活跃 |

> [!CAUTION]注意
> Tile级操作（如`pypto_pro.language.add`、`pypto_pro.language.mul`）内部会自动管理掩码，用户通过`pypto_pro.language.set_vec_mask`设置的掩码会被覆盖。这些API主要用于自定义操作或底层控制场景。

## 流水类型

S（标量流水）。

## 调用示例

下面是一个完整Kernel：在`pypto_pro.language.add`前调用`pypto_pro.language.set_vec_mask(-1, -1)`设置全1掩码，验证掩码状态未损坏。Vector Kernel开启`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def set_vec_mask_kernel(
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
        pl.set_vec_mask(-1, -1)
        pl.add(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```

其他典型用法（节选）：

```python
# norm 模式：只对前 32 个元素做计算
pl.set_vec_mask(0, 0xFFFFFFFF)

# 全元素不活跃
pl.set_vec_mask(0, 0)

# count 模式：设置有效元素个数（需先切换到 count 模式）
pl.set_mask_count()
pl.set_vec_mask(0, valid_cols)
```
