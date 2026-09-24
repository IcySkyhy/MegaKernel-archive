# pypto_pro.language.sort32

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

分组排序：把src每行按连续32个元素分组并按降序排序，同时跟踪输入的32位索引。输出tile保存8字节的val-idx记录；因此FP16输入的每个值占4个输出元素，FP32输入的每个值占2个输出元素。

对于不足32元素的尾块场景，需提供`tmp`参数作为中间缓冲。

## 函数原型

```python
pypto_pro.language.sort32(dst, src, idx, tmp=None)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst` | 输出 | 目标tile，存放排序后的val-idx对 |
| `src` | 输入 | 源tile（待排序的值） |
| `idx` | 输入 | 索引tile（原始位置索引） |
| `tmp` | 输入 | 可选，临时tile（尾块场景需要） |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst` | 输出 | Vec、行主序Tile，dtype与`src`相同且只能为FP16或FP32。每个输入值对应一个8字节val-idx记录，因此FP16时输出列数为输入有效列数的4倍，FP32时为2倍 |
| `src` | 输入 | Vec、行主序Tile，dtype为FP16或FP32；每行按32个值为一组排序，可包含一组或多组 |
| `idx` | 输入 | Vec、行主序Tile，dtype必须为`pypto_pro.language.DT_UINT32`；一行索引可由各数据行共用，多行索引则逐行使用 |
| `tmp` | 输入 | 可选Vec临时Tile。当`src.valid_shape`的列数不是32的整数倍时必须提供，用于处理最后一个不足32个值的分组 |

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整kernel：用`pypto_pro.language.sort32`对32个元素按降序排序并跟踪原始索引。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def sort32_kernel(
    a: pl.Tensor[[1, 32], pl.DT_FP16],
    idx_in: pl.Tensor[[1, 32], pl.DT_UINT32],
    sorted_out: pl.Tensor[[1, 128], pl.DT_FP16],
):
    tt_src = pl.TileType(shape=[1, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tt_dst = pl.TileType(shape=[1, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tt_idx = pl.TileType(shape=[1, 32], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    tile_src = pl.make_tile_group(type=tt_src, addrs=0x0000, mutex_ids=[0])
    tile_dst = pl.make_tile_group(type=tt_dst, addrs=0x0040, mutex_ids=[1])
    tile_idx = pl.make_tile_group(type=tt_idx, addrs=0x0140, mutex_ids=[2])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_dst = tile_dst.current()
        cur_idx = tile_idx.current()
        pl.load(cur_src, a, [0, 0])
        pl.load(cur_idx, idx_in, [0, 0])
        pl.sort32(cur_dst, cur_src, cur_idx)
        pl.store(sorted_out, cur_dst, [0, 0])
```

尾块场景（不足32元素，需要tile_tmp）：

```python
@pl.jit(auto_mutex=True)
def sort32_tail_kernel(
    a: pl.Tensor[[1, 16], pl.DT_FP16],
    idx_in: pl.Tensor[[1, 16], pl.DT_UINT32],
    sorted_out: pl.Tensor[[1, 64], pl.DT_FP16],
):
    tt_src = pl.TileType(shape=[1, 16], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tt_dst = pl.TileType(shape=[1, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tt_idx = pl.TileType(shape=[1, 16], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    tt_tmp = pl.TileType(shape=[1, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_src = pl.make_tile_group(type=tt_src, addrs=0x0000, mutex_ids=[0])
    tile_dst = pl.make_tile_group(type=tt_dst, addrs=0x0020, mutex_ids=[1])
    tile_idx = pl.make_tile_group(type=tt_idx, addrs=0x00A0, mutex_ids=[2])
    tile_tmp = pl.make_tile_group(type=tt_tmp, addrs=0x0120, mutex_ids=[3])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_dst = tile_dst.current()
        cur_idx = tile_idx.current()
        cur_tmp = tile_tmp.current()
        pl.load(cur_src, a, [0, 0])
        pl.load(cur_idx, idx_in, [0, 0])
        pl.sort32(cur_dst, cur_src, cur_idx, cur_tmp)
        pl.store(sorted_out, cur_dst, [0, 0])
```
