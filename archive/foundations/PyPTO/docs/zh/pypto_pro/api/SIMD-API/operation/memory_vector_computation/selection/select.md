# pypto_pro.language.select

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

按掩码在两个tile中逐元素选择，掩码为真取lhs，为假取rhs。常与`pypto_pro.language.eq/ne/lt/le/gt/ge`配合使用。

## 函数原型

```python
pypto_pro.language.select(out, mask, lhs, rhs, tmp)
```

> 5个参数均按函数原型中的顺序以位置参数传入。

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 选择结果tile |
| `mask` | 输入 | 掩码tile（由`eq`/`ne`/`lt`/`le`/`gt`/`ge`产生的bit-packed UINT8） |
| `lhs` | 输入 | mask为真时取的源tile |
| `rhs` | 输入 | mask为假时取的源tile或标量 |
| `tmp` | 输入 | 兼容性临时tile |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | tile-tile模式支持8/16/32/64位整型、FP16、BF16和FP32；tile-scalar模式支持8/16/32/64位整型、FP16和FP32<br>shape须与输入数据Tile一致 |
| `mask` | 输入 | 数据类型：UINT8（bit-packed掩码）<br>shape：与`lhs`一致<br>须由`pypto_pro.language.eq/ne/lt/le/gt/ge`产生，不能手动构造 |
| `lhs` | 输入 | 数据类型：与`out`一致<br>shape：与`out`一致 |
| `rhs` | 输入 | Tile：数据类型和shape与`out`一致；标量：须与`out`元素类型兼容 |
| `tmp` | 输入 | API必传的Vec Tile兼容性参数；当前实现不读写该参数，不要求与`out`同类型或同shape |

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整kernel：用`pypto_pro.language.gt`生成bit-packed谓词后，`pypto_pro.language.select`按谓词选择两个FP32 tile中的一个写回GM。掩码为真取`lhs`，为假取`rhs`。纯vector kernel使用`make_tile_group`管理Tile资源，并通过`auto_mutex`完成Tile访问间的同步。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def scalar_gt_select_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP32],
    b: pl.Tensor[[64, 128], pl.DT_FP32],
    mask_in: pl.Tensor[[64, 128], pl.DT_FP16],
    out: pl.Tensor[[64, 128], pl.DT_FP32],
):
    tt32 = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a_group = pl.make_tile_group(type=tt32, addrs=0x0000, mutex_ids=[0])
    tile_b_group = pl.make_tile_group(type=tt32, addrs=0x8000, mutex_ids=[1])
    tile_out_group = pl.make_tile_group(type=tt32, addrs=0x10000, mutex_ids=[2])
    tmp_vec_group = pl.make_tile_group(type=tt32, addrs=0x18000, mutex_ids=[3])
    mask_fp16_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x20000, mutex_ids=[4])
    mask_vec_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
        addrs=0x24000, mutex_ids=[5])
    with pl.section_vector():
        tile_a = tile_a_group.current()
        tile_b = tile_b_group.current()
        tile_out = tile_out_group.current()
        tmp_vec = tmp_vec_group.current()
        mask_fp16 = mask_fp16_group.current()
        mask_vec = mask_vec_group.current()
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.load(mask_fp16, mask_in, [0, 0])
        # mask_fp16 > 0 -> bit-packed 谓词 mask_vec（cmp_mode=4 为 gt）
        pl.gt(mask_vec, mask_fp16, 0.0)
        # 谓词为真取 lhs(=a)，否则取 rhs(=b)
        pl.select(tile_out, mask_vec, tile_a, tile_b, tmp_vec)
        pl.store(out, tile_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:select:start -->
```bash
输入数据a：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [65 65.25 65.5 65.75 66 66.25 66.5 66.75 ...], [97 97.25 97.5 97.75 98 98.25 98.5 98.75 ...], ...]
输入数据b：[[8 7.875 7.75 7.625 7.5 7.375 7.25 7.125 ...], [-8 -8.125 -8.25 -8.375 -8.5 -8.625 -8.75 -8.875 ...], [-24 -24.125 -24.25 -24.375 -24.5 -24.625 -24.75 -24.875 ...], [-40 -40.125 -40.25 -40.375 -40.5 -40.625 -40.75 -40.875 ...], ...]
输入数据mask：[[1 -1 1 -1 1 -1 1 -1 ...], [1 -1 1 -1 1 -1 1 -1 ...], [1 -1 1 -1 1 -1 1 -1 ...], [1 -1 1 -1 1 -1 1 -1 ...], ...]
输出数据out：[[1 7.875 1.5 7.625 2 7.375 2.5 7.125 ...], [33 -8.125 33.5 -8.375 34 -8.625 34.5 -8.875 ...], [65 -24.125 65.5 -24.375 66 -24.625 66.5 -24.875 ...], [97 -40.125 97.5 -40.375 98 -40.625 98.5 -40.875 ...], ...]
```
<!-- pypto-doc-output:select:end -->
