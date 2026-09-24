# Tile subview（切片子视图）

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

对已加载到UB的Tile执行二维切片，生成一个带偏移和有效shape的子Tile（sub-view）。子Tile共享原Tile的UB缓冲区，不分配新内存。

使用方式分两种：

- **Cube/Vector section**：使用切片语法`tile[r:r2, c:c2]`，生成new Tile，自动设置偏移和`SetValidShape`。
- **vector_function**：使用指针偏移`tile + offset`，仅做UB指针运算，不生成Tile描述符。

> [!CAUTION]注意
> Tensor不支持切片语法，请使用`pl.load`/`pl.store`配合offset列表访问。

## 语法

### 子Tile切片

```python
sub = tile[row_start:row_stop, col_start:col_stop]
```

切片遵循Python半开区间语义：`[row_start, row_stop)`、`[col_start, col_stop)`。

- 缺省`row_stop` / `col_stop`时取到Tile shape对应维度末尾。
- 子Tile保留原Tile的物理shape（row_stride不变），通过`SetValidShape`设置有效数据范围。
- 若原Tile设置了`valid_shape`（编译期或运行时`pl.set_validshape`），切片会与之求交集clamp。

### VF指针偏移

```python
ptr = tile + offset    # offset 为线性元素偏移
```

VF段内`tile + offset`退化为纯指针偏移`(vf_tile_ptr_N + offset)`，shape / valid_shape信息被忽略。适用于`vf.load_align` / `vf.store_align`等基于指针的VF操作。

> **VF段不建议使用切片语法**`tile[r:r2, c:c2]`。虽然技术上可用（会warning并退化为指针偏移），但`tile + offset`语义更直接。

## 参数说明

### 子Tile切片`tile[row_start:row_stop, col_start:col_stop]`

| 参数                 | 输入/输出 | 说明                                       |
| -------------------- | --------- | ------------------------------------------ |
| `tile`               | 输入      | 源Tile，须为2D TileType                  |
| `row_start:row_stop` | 输入      | 行切片，半开区间，整型常量或运行时整型标量表达式    |
| `col_start:col_stop` | 输入      | 列切片，半开区间，整型常量或运行时整型标量表达式    |
| 返回值               | 输出      | 子Tile（TileType），共享原Tile UB缓冲区 |

### VF段偏移`tile + offset`

| 参数     | 输入/输出 | 说明                                                          |
| -------- | --------- | ------------------------------------------------------------- |
| `tile`   | 输入      | 源Tile，须为TileType                                        |
| `offset` | 输入      | 线性元素偏移（整型常量或运行时整型标量表达式）                         |
| 返回值   | 输出      | 偏移后的指针表达式，供`vf.load_align` / `vf.store_align`使用 |

## 调用示例

### 基本切片

从`[8, 64]`的Tile中取`[2:6, 16:50]`子块（4行 × 34列），直接store到GM。子Tile的row_stride仍为64（原Tile列数），store时按valid_shape `[4, 34]`搬运。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def tile_subview_kernel(
    a: pl.Tensor[[8, 64], pl.DT_FP16],
    out: pl.Tensor[[4, 34], pl.DT_FP16],
):
    tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[8, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[0],
    )
    with pl.section_vector():
        tile = tile_group.next()
        pl.load(tile, a, [0, 0])
        sub = tile[2:6, 16:50]
        pl.store(out, sub, [0, 0])
```

### 与set_validshape交集

Tile shape `[8, 64]`，运行时设置`valid_shape=[6, 40]`，切片`[2:6, 16:50]`会被clamp到`[4, 24]`（`min(4, 6-2)` × `min(34, 40-16)`）。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def tile_subview_vshape_kernel(
    a: pl.Tensor[[8, 64], pl.DT_FP16],
    out1: pl.Tensor[[4, 24], pl.DT_FP16],
):
    tile_group = pl.make_tile_group(
        type=pl.TileType(shape=[8, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1]),
        addrs=0x0000, mutex_ids=[0],
    )
    with pl.section_vector():
        tile = tile_group.next()
        pl.set_validshape(tile, [6, 40])
        pl.load(tile, a, [0, 0])
        sub1 = tile[2:6, 16:50]
        pl.store(out1, sub1, [0, 0])
```

### VF段：指针偏移搬运

在`@pl.vector_function`内部，使用`tile + offset`做指针偏移，配合`vf.load_align` / `vf.store_align`逐行搬运子矩阵。

```python
import pypto_pro.language as pl


@pl.vector_function
def vf_copy_subview_row(dst_tile, src_tile, n_rows, n_cols, stride):
    preg = vf.update_mask(n_cols, dtype=pl.DT_FP16)
    for r in pl.range(n_rows):
        vreg = vf.load_align(src_tile, r * stride)
        vf.store_align(dst_tile + r * n_cols, vreg, preg)


@pl.jit(auto_mutex=True)
def tile_subview_vf_kernel(
    vf_a: pl.Tensor[[4, 64], pl.DT_FP16],
    vf_out: pl.Tensor[[3, 32], pl.DT_FP16],
):
    vf_src_group = pl.make_tile_group(
        type=pl.TileType(shape=[4, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000, mutex_ids=[1],
    )
    vf_dst_group = pl.make_tile_group(
        type=pl.TileType(shape=[3, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x5000, mutex_ids=[2],
    )
    with pl.section_vector():
        vf_src = vf_src_group.next()
        vf_dst = vf_dst_group.next()
        pl.load(vf_src, vf_a, [0, 0])
        # 在 block 段用切片语法生成 subview，传入 VF 函数后作为偏移指针使用
        vf_copy_subview_row(vf_dst, vf_src[1:4, 16:48], 3, 32, 64)
        pl.store(vf_out, vf_dst, [0, 0])
```

> [!NOTE]说明
> `vf_src[1:4, 16:48]`在block段（`section_vector`内、VF函数外）使用切片语法，生成带偏移的子Tile。传入VF函数后，VF段通过`tile + offset`形式做指针偏移搬运。VF函数内部的`dst_tile + r * n_cols`是VF段指针偏移。
