# pypto_pro.language.mrgsort

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

归并排序：对src tile中的元素按块进行归并排序，结果写入dst tile。输入数据为val-idx对格式（每个元素包含值和索引），排序后保持索引与值的对应关系。

典型场景：TopK排序的预处理步骤，先对每个块内部排序，再用[`pypto_pro.language.mrgsort2`](mrgsort2.md)多路归并。

## 函数原型

```python
pypto_pro.language.mrgsort(dst, src, *, block_len)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst` | 输出 | 目标tile，存放排序结果 |
| `src` | 输入 | 源tile，val-idx对格式 |
| `block_len` | 输入 | 归并块长度 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst` | 输出 | 数据类型：与`src`一致<br>shape：与`src`一致 |
| `src` | 输入 | 数据类型：b16、b32<br>shape：行数为1<br>数据格式：val-idx对（每个元素含值和原始索引） |
| `block_len` | 输入 | 编译期正整数，单位为Tile存储元素，并同时包含value和index。例如32个FP32 value-index记录对应64个FP32存储元素 |

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整kernel：用`pypto_pro.language.mrgsort`对UB上的val-idx对数据做归并排序。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def mrgsort_kernel(
    a: pl.Tensor[[1, 1024], pl.DT_FP16],
    sorted_out: pl.Tensor[[1, 1024], pl.DT_FP16],
):
    tt = pl.TileType(shape=[1, 1024], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_src = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_dst = pl.make_tile_group(type=tt, addrs=0x0800, mutex_ids=[1])
    with pl.section_vector():
        cur_src = tile_src.current()
        cur_dst = tile_dst.current()
        pl.load(cur_src, a, [0, 0])
        pl.mrgsort(cur_dst, cur_src, block_len=256)
        pl.store(sorted_out, cur_dst, [0, 0])
```
