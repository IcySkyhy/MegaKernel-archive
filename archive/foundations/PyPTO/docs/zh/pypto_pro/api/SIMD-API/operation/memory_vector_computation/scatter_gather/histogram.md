# pypto_pro.language.histogram

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

直方图统计：对源tile中的元素按字节值进行计数，结果写入目标tile。用于基数排序（radix sort）中统计每个桶的元素个数。

通过`is_msb`参数控制统计高字节还是低字节：

- `is_msb=True`：统计每个元素的最高字节（UINT16为bits 15-8，UINT32为bits 31-24）
- `is_msb=False`：统计每个元素的低字节（bits 7-0）；仅纳入高字节与`idx` tile中对应行值匹配的元素

## 函数原型

```python
pypto_pro.language.histogram(dst, src, idx, *, is_msb)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst` | 输出 | 目标tile，存放直方图统计结果 |
| `src` | 输入 | 源tile，待统计的元素 |
| `idx` | 输入 | 索引tile，`is_msb=False`时用于过滤 |
| `is_msb` | 输入 | 是否统计高字节 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst` | 输出 | 数据类型：`pypto_pro.language.DT_UINT32`<br>shape：行数与`src`一致，列数 ≥ 256（覆盖所有可能的字节值）<br>布局：row_major + none_box（`layout=pl.ND`） |
| `src` | 输入 | 数据类型：`pypto_pro.language.DT_UINT16`或`pypto_pro.language.DT_UINT32`<br>shape：任意<br>布局：row_major + none_box（`layout=pl.ND`） |
| `idx` | 输入 | 数据类型：`pypto_pro.language.DT_UINT8`<br>`src`为`DT_UINT16`时：shape行数与`src`一致，列数为1，DN布局（`layout=pl.DN`）<br>`src`为`DT_UINT32`且`is_msb=True`时：`idx`不使用<br>`src`为`DT_UINT32`且`is_msb=False`时：shape行数为3、列数与`src`一致，ND布局（`layout=pl.ND`） |
| `is_msb` | 输入 | `True`或`False` |

## 流水类型

V（向量计算流水）。

## 调用示例

```python
import pypto_pro.language as pl

ROWS = 32
COLS = 128
IDX_COLS_DN = 1


@pl.jit(auto_mutex=True)
def histogram_kernel(
    src: pl.Tensor[[ROWS, COLS], pl.DT_UINT16],
    idx: pl.Tensor[[ROWS, IDX_COLS_DN], pl.DT_UINT8],
    out: pl.Tensor[[ROWS, 256], pl.DT_UINT32],
):
    pl.system.bar_all()
    tt_src = pl.TileType(shape=[ROWS, COLS], dtype=pl.DT_UINT16,
                         target_memory=pl.MemorySpace.Vec, layout=pl.ND)
    tt_idx = pl.TileType(shape=[ROWS, IDX_COLS_DN], dtype=pl.DT_UINT8,
                         target_memory=pl.MemorySpace.Vec, layout=pl.DN)
    tt_dst = pl.TileType(shape=[ROWS, 256], dtype=pl.DT_UINT32,
                         target_memory=pl.MemorySpace.Vec, layout=pl.ND)
    tile_src = pl.make_tile_group(type=tt_src, addrs=0x0000, mutex_ids=[0])
    tile_idx = pl.make_tile_group(type=tt_idx, addrs=0x2000, mutex_ids=[1])
    tile_dst = pl.make_tile_group(type=tt_dst, addrs=0x2020, mutex_ids=[2])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_idx = tile_idx.current()
        cur_dst = tile_dst.current()
        pl.load(cur_src, src, [0, 0])
        pl.load(cur_idx, idx, [0, 0])
        pl.histogram(cur_dst, cur_src, cur_idx, is_msb=True)
        pl.store(out, cur_dst, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:histogram:start -->
```bash
输入数据src：[[0 257 514 771 1028 1285 1542 1799 ...], [32896 33153 33410 33667 33924 34181 34438 34695 ...], [256 513 770 1027 1284 1541 1798 2055 ...], [33152 33409 33666 33923 34180 34437 34694 34951 ...], ...]
输入数据idx：[[0], [0], [0], [0], ...]
输出数据out：[[1 2 3 4 5 6 7 8 ...], [0 0 0 0 0 0 0 0 ...], [0 1 2 3 4 5 6 7 ...], [1 1 1 1 1 1 1 1 ...], ...]
```
<!-- pypto-doc-output:histogram:end -->
