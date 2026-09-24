# pypto_pro.language.argmin

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

沿指定维度查找`src` tile的最小元素索引并写入`out`。`dim=0`沿最后一维查找每行最小元素的列索引；`dim=1`沿第一维查找每列最小元素的行索引。

## 函数原型

```python
pypto_pro.language.argmin(out, src, tmp, *, dim=0)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 目标tile，存放每行最小元素的列索引 |
| `src` | 输入 | 源tile |
| `tmp` | 输入 | 临时tile（硬件中间计算用） |
| `dim` | 输入 | 查找维度 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 数据类型：`DT_INT32`或`DT_UINT32`<br>`dim=0`时shape为`[行数, 1]`；`dim=1`时shape为`[1, 列数]` |
| `src` | 输入 | `dim=0`支持FP16和FP32；`dim=1`支持8/16/32位整型或FP16/FP32/BF16<br>shape：二维Vec Tile |
| `tmp` | 输入 | 公共接口必传；数据类型、shape和valid_shape与`src`一致。`dim=0`后端不消费该临时Tile，但仍须按函数原型提供 |
| `dim` | 输入 | `0`：返回每行最小元素的列索引；`1`：返回每列最小元素的行索引。默认值为`0` |

## 流水类型

V（向量计算流水）。

## 调用示例

### dim = 0

下面是一个完整kernel：对64×128 FP16源tile做行向取最小值索引，输出`[64, 1]` INT32。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl

M, N = 64, 128


@pl.jit(auto_mutex=True)
def row_argmin_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    z: pl.Tensor[[M, 1], pl.DT_INT32],
):
    tile_a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[0])
    tile_tmp = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(
        type=pl.TileType(shape=[64, 1], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.argmin(cur_out, cur_a, cur_tmp)
        pl.store(z, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:row_argmin:start -->
```bash
输入数据a：[[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
输出数据z：[[0], [0], [0], [0], ...]
```
<!-- pypto-doc-output:row_argmin:end -->

### dim = 1

下面的kernel对64×128 FP16源tile做列向查找，输出`[1, 128]` INT32。

```python
M, N = 64, 128

@pl.jit(auto_mutex=True)
def col_argmin_kernel(
    x: pl.Tensor[[M, N], pl.DT_FP16],
    z: pl.Tensor[[1, N], pl.DT_INT32],
):
    tile_a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[0])
    tile_tmp = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(
        type=pl.TileType(shape=[1, 128], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
        addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, x, [0, 0])
        pl.argmin(cur_out, cur_a, cur_tmp, dim=1)
        pl.store(z, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:col_argmin:start -->
```bash
输入数据a：[[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
输出数据z：[[0 0 0 0 0 0 0 0 ...]]
```
<!-- pypto-doc-output:col_argmin:end -->
