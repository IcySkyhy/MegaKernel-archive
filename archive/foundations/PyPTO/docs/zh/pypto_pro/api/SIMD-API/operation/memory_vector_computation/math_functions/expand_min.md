# pypto_pro.language.expand_min

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

将指定维度的单元素tile广播到`src`的shape后逐元素取较小值。`dim=0`广播`[行数, 1]` tile；`dim=1`广播`[1, 列数]` tile。

## 函数原型

```python
pypto_pro.language.expand_min(out, src, scalar, *, dim=0)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 目标tile |
| `src` | 输入 | 源tile |
| `scalar` | 输入 | `[行数, 1]` tile，广播到每列 |
| `dim` | 输入 | 展开方向 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 数据类型：与`src`一致<br>shape：与`src`一致 |
| `src` | 输入 | 数据类型：8/16/32位整型、FP16、BF16和FP32<br>shape：二维Vec Tile |
| `scalar` | 输入 | 数据类型：与`src`一致<br>`dim=0`时shape为`[行数, 1]`；`dim=1`时shape为`[1, 列数]` |
| `dim` | 输入 | `0`：广播`[行数, 1]` tile；`1`：广播`[1, 列数]` tile。默认值为`0` |

## 流水类型

V（向量计算流水）。

## 调用示例

### dim = 0

下面是一个完整kernel：对64×128 FP16源tile与`[64, 1]`行向量做行向最小值展开，输出`[64, 128]`。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl

M, N = 64, 128


@pl.jit(auto_mutex=True)
def row_expand_min_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    y: pl.Tensor[[M, 1], pl.DT_FP16],
    z: pl.Tensor[[M, N], pl.DT_FP16],
):
    tile_a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[0])
    tile_row = pl.make_tile_group(
        type=pl.TileType(shape=[64, 1], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_row = tile_row.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_row, y, [0, 0])
        pl.expand_min(cur_out, cur_a, cur_row)
        pl.store(z, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:row_expand_min:start -->
```bash
输入数据a：[[-3 -2.875 -2.75 -2.625 -2.5 -2.375 -2.25 -2.125 ...], [13 13.125 13.25 13.375 13.5 13.625 13.75 13.875 ...], [29 29.125 29.25 29.375 29.5 29.625 29.75 29.875 ...], [45 45.125 45.25 45.375 45.5 45.625 45.75 45.875 ...], ...]
输入数据v：[[1.25], [1.5], [1.75], [2], ...]
输出数据z：[[-3 -2.875 -2.75 -2.625 -2.5 -2.375 -2.25 -2.125 ...], [1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5 ...], [1.75 1.75 1.75 1.75 1.75 1.75 1.75 1.75 ...], [2 2 2 2 2 2 2 2 ...], ...]
```
<!-- pypto-doc-output:row_expand_min:end -->

### dim = 1

```python
import pypto_pro.language as pl

M, N = 64, 128


@pl.jit(auto_mutex=True)
def col_expand_min_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    y: pl.Tensor[[1, N], pl.DT_FP16],
    z: pl.Tensor[[M, N], pl.DT_FP16],
):
    tile_a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[0])
    tile_col = pl.make_tile_group(
        type=pl.TileType(shape=[1, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_col = tile_col.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_col, y, [0, 0])
        pl.expand_min(cur_out, cur_a, cur_col, dim=1)
        pl.store(z, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:col_expand_min:start -->
```bash
输入数据a：[[-3 -2.875 -2.75 -2.625 -2.5 -2.375 -2.25 -2.125 ...], [13 13.125 13.25 13.375 13.5 13.625 13.75 13.875 ...], [29 29.125 29.25 29.375 29.5 29.625 29.75 29.875 ...], [45 45.125 45.25 45.375 45.5 45.625 45.75 45.875 ...], ...]
输入数据v：[[1.25 1.5 1.75 2 2.25 2.5 2.75 3 ...]]
输出数据z：[[-3 -2.875 -2.75 -2.625 -2.5 -2.375 -2.25 -2.125 ...], [1.25 1.5 1.75 2 2.25 2.5 2.75 3 ...], [1.25 1.5 1.75 2 2.25 2.5 2.75 3 ...], [1.25 1.5 1.75 2 2.25 2.5 2.75 3 ...], ...]
```
<!-- pypto-doc-output:col_expand_min:end -->
