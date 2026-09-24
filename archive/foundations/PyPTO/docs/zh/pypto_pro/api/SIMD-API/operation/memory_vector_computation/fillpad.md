# pypto_pro.language.fillpad

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

填充Tile的padding区域。当Tile设置的`valid_shape`（有效形状）小于`shape`（物理形状）时，有效区域之外的部分为padding区域。`fillpad`根据Tile创建时指定的`pad`值（如`pypto_pro.language.TilePad.zero`）填充该区域。

`mode`用于选择填充模式：

- `NORMAL`：`out`和`src`的shape相同、地址不同。
- `EXPAND`：允许`out`的shape大于`src`的shape，将源Tile的有效数据复制到目标Tile，并填充扩展区域。
- `INPLACE`：`out`和`src`的shape相同、地址相同，直接在原地址上填充。

## 函数原型

```python
pypto_pro.language.fillpad(out, src, *, mode=pypto_pro.language.FillPadMode.NORMAL)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 目标Tile，padding区域将被填充 |
| `src` | 输入 | 源Tile，提供有效数据和有效形状信息 |
| `mode` | 输入 | 填充模式，默认`pypto_pro.language.FillPadMode.NORMAL` |

## 参数范围

### NORMAL模式

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 数据类型：b8、b16、b32<br>shape须与`src`一致<br>必须设置非`null`的`pad`属性（如`pypto_pro.language.TilePad.zero`）<br>地址须与`src`不同 |
| `src` | 输入 | 数据类型：b8、b16、b32，数据位宽须与`out`一致<br>shape须与`out`一致<br>padding范围由`src`的有效形状确定；可在TileType中指定，或通过`set_validshape`设置 |
| `mode` | 输入 | `pypto_pro.language.FillPadMode.NORMAL`，可省略 |

### EXPAND模式

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 数据类型：b8、b16、b32<br>shape各维度须大于等于`src`<br>必须设置非`null`的`pad`属性（如`pypto_pro.language.TilePad.zero`）<br>地址须与`src`不同 |
| `src` | 输入 | 数据类型：b8、b16、b32，数据位宽须与`out`一致<br>shape各维度须小于等于`out`<br>padding范围由`src`的有效形状确定；可在TileType中指定，或通过`set_validshape`设置 |
| `mode` | 输入 | `pypto_pro.language.FillPadMode.EXPAND` |

### INPLACE模式

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 数据类型：b8、b16、b32<br>shape须与`src`一致<br>必须设置非`null`的`pad`属性（如`pypto_pro.language.TilePad.zero`）<br>地址须与`src`相同 |
| `src` | 输入 | 数据类型：b8、b16、b32，数据位宽须与`out`一致<br>shape须与`out`一致<br>padding范围由`src`的有效形状确定；可在TileType中指定，或通过`set_validshape`设置 |
| `mode` | 输入 | `pypto_pro.language.FillPadMode.INPLACE` |

## 补充说明

`fillpad`支持Vec和Mat Tile，源、目标必须位于相同内存空间。Mat上的`NORMAL`模式要求源、目标具有完全相同的Tile类型（包括shape、valid_shape和pad）；Mat上的异构`EXPAND`跨层接口尚未定型，不应使用。`INPLACE`还要求源、目标共享同一内存地址。

## 流水类型

V（向量计算流水）。

## 调用示例

### NORMAL模式

本例中，源Tile和目标Tile的物理shape均为`[8, 8]`，但使用不同地址。源Tile的有效shape设置为`[5, 7]`后，目标Tile的`[0:5, 0:7]`区域保留源数据，其余区域补零。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def fillpad_kernel(
    x: pl.Tensor[[8, 8], pl.DT_INT32],
    z: pl.Tensor[[8, 8], pl.DT_INT32],
):
    src_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32,
                           target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    dst_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32,
                           target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.zero)
    src = pl.make_tile_group(type=src_type, addrs=0x0000, mutex_ids=[0])
    dst = pl.make_tile_group(type=dst_type, addrs=0x0100, mutex_ids=[1])
    with pl.section_vector():
        cur_src = src.current()
        cur_dst = dst.current()
        pl.set_validshape(cur_src, [5, 7])
        pl.load(cur_src, x, [0, 0])
        pl.fillpad(cur_dst, cur_src)
        pl.store(z, cur_dst, [0, 0])
```

### EXPAND模式

本例中，源Tile的物理shape为`[8, 8]`、有效shape为`[5, 7]`，目标Tile的物理shape扩展为`[8, 16]`，并使用不同地址。执行后，目标Tile的`[0:5, 0:7]`区域保留源数据，其余区域补零。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def fillpad_expand_kernel(
    x: pl.Tensor[[8, 8], pl.DT_INT32],
    z: pl.Tensor[[8, 16], pl.DT_INT32],
):
    src_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32,
                           target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    dst_type = pl.TileType(shape=[8, 16], dtype=pl.DT_INT32,
                           target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.zero)
    src = pl.make_tile_group(type=src_type, addrs=0x0000, mutex_ids=[0])
    dst = pl.make_tile_group(type=dst_type, addrs=0x0100, mutex_ids=[1])
    with pl.section_vector():
        cur_src = src.current()
        cur_dst = dst.current()
        pl.set_validshape(cur_src, [5, 7])
        pl.load(cur_src, x, [0, 0])
        pl.fillpad(cur_dst, cur_src, mode=pl.FillPadMode.EXPAND)
        pl.store(z, cur_dst, [0, 0])
```

### INPLACE模式

本例中，源Tile和目标Tile的物理shape均为`[8, 8]`，并使用同一地址`0x0000`。源Tile的有效shape设置为`[5, 7]`后，`INPLACE`直接在原地址上将其余区域补零，不占用额外地址空间。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def fillpad_inplace_kernel(
    x: pl.Tensor[[8, 8], pl.DT_INT32],
    z: pl.Tensor[[8, 8], pl.DT_INT32],
):
    src_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32,
                           target_memory=pl.MemorySpace.Vec,
                           pad=pl.TilePad.zero, valid_shape=[-1, -1])
    dst_type = pl.TileType(shape=[8, 8], dtype=pl.DT_INT32,
                           target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.zero)
    src = pl.make_tile_group(type=src_type, addrs=0x0000, mutex_ids=[0])
    dst = pl.make_tile_group(type=dst_type, addrs=0x0000, mutex_ids=[1])
    with pl.section_vector():
        cur_src = src.current()
        cur_dst = dst.current()
        pl.set_validshape(cur_src, [5, 7])
        pl.load(cur_src, x, [0, 0])
        pl.fillpad(cur_dst, cur_src, mode=pl.FillPadMode.INPLACE)
        pl.store(z, cur_dst, [0, 0])
```
