# pypto_pro.language.axpy

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

向量标量乘加：`out[i] = alpha * src[i] + out[i]`。将源Tile每个元素乘以标量alpha后累加到目标Tile。

## 函数原型

```python
pypto_pro.language.axpy(out, src, alpha)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输入/输出 | 目标Tile，同时作为累加器输入和输出 |
| `src` | 输入 | 源Tile，乘以`alpha`后累加到`out` |
| `alpha` | 输入 | 标量乘数 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输入/输出 | Vec Tile，dtype支持FP16或FP32；shape和`valid_shape`须与`src`一致。通常dtype与`src`相同；另支持`src`为FP16、`out`为FP32的拓宽计算。运算前须已载入累加器初值，运算后被结果覆盖 |
| `src` | 输入 | Vec Tile，dtype支持FP16或FP32；shape和`valid_shape`须与`out`一致 |
| `alpha` | 输入 | 编译期常量或运行时浮点标量表达式；标量dtype必须与`src`元素dtype完全一致 |

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整Kernel：载入两个FP32 Tile，用`pypto_pro.language.axpy`做`y = 2.0 * x + y`融合标量乘加。纯Vector Kernel使用`make_tile_group`管理Tile资源，并通过`auto_mutex`完成流水同步。

```python
import pypto_pro.language as pl

ALPHA = 2.0


@pl.jit(auto_mutex=True)
def axpy_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    y: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_x_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_y_group = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        tile_x = tile_x_group.current()
        tile_y = tile_y_group.current()
        pl.load(tile_x, x, [0, 0])
        pl.load(tile_y, y, [0, 0])
        pl.axpy(tile_y, tile_x, ALPHA)
        pl.store(y, tile_y, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:axpy:start -->
```bash
输入数据x：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [17 17.25 17.5 17.75 18 18.25 18.5 18.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [49 49.25 49.5 49.75 50 50.25 50.5 50.75 ...], ...]
输入数据y原始值：[[4 3.875 3.75 3.625 3.5 3.375 3.25 3.125 ...], [-4 -4.125 -4.25 -4.375 -4.5 -4.625 -4.75 -4.875 ...], [-12 -12.125 -12.25 -12.375 -12.5 -12.625 -12.75 -12.875 ...], [-20 -20.125 -20.25 -20.375 -20.5 -20.625 -20.75 -20.875 ...], ...]
输出数据y：[[6 6.375 6.75 7.125 7.5 7.875 8.25 8.625 ...], [30 30.375 30.75 31.125 31.5 31.875 32.25 32.625 ...], [54 54.375 54.75 55.125 55.5 55.875 56.25 56.625 ...], [78 78.375 78.75 79.125 79.5 79.875 80.25 80.625 ...], ...]
```
<!-- pypto-doc-output:axpy:end -->
