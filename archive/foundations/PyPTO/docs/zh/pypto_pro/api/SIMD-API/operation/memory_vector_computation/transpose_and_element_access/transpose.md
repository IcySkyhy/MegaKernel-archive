# pypto_pro.language.transpose

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

交换二维tile的两个轴，实现矩阵转置。

## 函数原型

```python
pypto_pro.language.transpose(out, src)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 转置结果tile |
| `src` | 输入 | 源tile |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 数据类型：与`src`一致<br>shape：为`src`转置后的结果（如`src`为`[64, 128]`时`out`为`[128, 64]`）<br>不可与`src`为同一tile |
| `src` | 输入 | 数据类型：8/16/32位整型、FP16、BF16或FP32，不支持64位元素<br>shape：二维Vec Tile；源、目标主维长度乘元素字节数须32字节对齐 |

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整kernel：把64×64 FP16源tile转置后写回GM。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def transpose_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.transpose(cur_out, cur_a)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:transpose:start -->
```bash
输入数据a：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [17 17.25 17.5 17.75 18 18.25 18.5 18.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [49 49.25 49.5 49.75 50 50.25 50.5 50.75 ...], ...]
输出数据out：[[1 17 33 49 65 81 97 113 ...], [1.25 17.25 33.25 49.25 65.25 81.25 97.25 113.25 ...], [1.5 17.5 33.5 49.5 65.5 81.5 97.5 113.5 ...], [1.75 17.75 33.75 49.75 65.75 81.75 97.75 113.75 ...], ...]
```
<!-- pypto-doc-output:transpose:end -->
