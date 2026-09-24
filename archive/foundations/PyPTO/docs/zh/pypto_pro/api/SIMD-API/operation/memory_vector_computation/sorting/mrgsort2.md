# pypto_pro.language.mrgsort2

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

多路归并排序：将2到4个已排序的源tile归并为一个有序输出。每个源tile内部已按降序排列（val-idx对格式），`mrgsort2`从中选取最大值依次写入dst。

`exhausted`控制硬件在某个输入列表耗尽时是否暂停归并，用于多步归并中处理长度不一致的源；它是整次调用的控制位，不用于指定某一个源。

## 函数原型

```python
pypto_pro.language.mrgsort2(src0, src1, dst, tmp, *extra_srcs, exhausted=False)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `src0` | 输入 | 第一个已排序源tile |
| `src1` | 输入 | 第二个已排序源tile |
| `dst` | 输出 | 目标tile，存放归并结果 |
| `tmp` | 输入 | 临时tile（硬件中间计算用） |
| `*extra_srcs` | 输入 | 可选，额外的已排序源tile（支持3路或4路归并） |
| `exhausted` | 输入 | 是否有源已耗尽 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `src0`, `src1` | 输入 | Vec、行主序、单行Tile；dtype为FP16或FP32，且所有源、`dst`和`tmp`必须一致；内部已按降序排列的8字节val-idx记录 |
| `dst` | 输出 | Vec、行主序、单行Tile；dtype与所有源一致。输出长度由`dst.valid_shape`控制，不要求物理shape与每个源完全相同 |
| `tmp` | 输入 | Vec、行主序、单行临时Tile；dtype与源一致，列数不得小于`dst`列数，并须满足所有输入列表合并所需的临时空间 |
| `*extra_srcs` | 输入 | 可选的第3、4个源Tile，格式同上；源总数只能为2～4个 |
| `exhausted` | 输入 | 编译期`bool`，默认`False`；`True`表示启用“任一输入列表耗尽时暂停”的硬件模式 |

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整kernel：用`pypto_pro.language.mrgsort2`把两个已排序的源tile归并为一个有序输出。纯vector kernel使用`make_tile_group`管理Tile资源，并通过`auto_mutex`完成流水同步。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def mrgsort2_kernel(
    src0_tensor: pl.Tensor[[1, 256], pl.DT_FP32],
    src1_tensor: pl.Tensor[[1, 256], pl.DT_FP32],
    sorted_out: pl.Tensor[[1, 256], pl.DT_FP32],
):
    tt = pl.TileType(shape=[1, 256], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_src0_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_src1_group = pl.make_tile_group(type=tt, addrs=0x0400, mutex_ids=[1])
    tile_dst_group = pl.make_tile_group(type=tt, addrs=0x0800, mutex_ids=[2])
    tile_tmp_group = pl.make_tile_group(type=tt, addrs=0x0C00, mutex_ids=[3])
    with pl.section_vector():
        tile_src0 = tile_src0_group.current()
        tile_src1 = tile_src1_group.current()
        tile_dst = tile_dst_group.current()
        tile_tmp = tile_tmp_group.current()
        pl.load(tile_src0, src0_tensor, [0, 0])
        pl.load(tile_src1, src1_tensor, [0, 0])
        pl.mrgsort2(tile_src0, tile_src1, tile_dst, tile_tmp, exhausted=False)
        pl.store(sorted_out, tile_dst, [0, 0])
```

其他典型用法（节选）：

```python
# 3 路归并
pl.mrgsort2(tile_src0, tile_src1, tile_dst, tile_tmp, tile_src2, exhausted=False)

# 4 路归并
pl.mrgsort2(tile_src0, tile_src1, tile_dst, tile_tmp, tile_src2, tile_src3, exhausted=False)
```
