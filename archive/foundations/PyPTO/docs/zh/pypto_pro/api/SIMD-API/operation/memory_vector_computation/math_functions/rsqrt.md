# pypto_pro.language.rsqrt

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

逐元素计算倒数平方根：`out = 1 / sqrt(src)`。支持in-place写法。

## 函数原型

```python
pypto_pro.language.rsqrt(out, src)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 目标tile，存放逐元素倒数平方根结果 |
| `src` | 输入 | 源tile |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 数据类型：`DT_FP16`、`DT_FP32`，须与`src`一致<br>内存空间：`MemorySpace.Vec`（UB）<br>layout：RowMajor（`pl.ND`）<br>shape和`valid_shape`须与`src`一致<br>支持与`src`为同一tile，实现in-place rsqrt |
| `src` | 输入 | 数据类型、内存空间、layout、shape和`valid_shape`：与`out`一致<br>元素值应大于0；若为0或负数，硬件结果未定义 |

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整kernel：把FP32源tile逐元素计算倒数平方根后写回GM。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def rsqrt_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32],
                 out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.rsqrt(cur_out, cur_a)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:rsqrt:start -->
```bash
输入数据a：[[1 1.125 1.25 1.375 1.5 1.625 1.75 1.875 ...], [9 9.125 9.25 9.375 9.5 9.625 9.75 9.875 ...], [17 17.125 17.25 17.375 17.5 17.625 17.75 17.875 ...], [25 25.125 25.25 25.375 25.5 25.625 25.75 25.875 ...], ...]
输出数据out：[[1 0.942809 0.894427 0.852803 0.816497 0.784465 0.755929 0.730297 ...], [0.333333 0.331042 0.328798 0.326599 0.324443 0.322329 0.320256 0.318223 ...], [0.242536 0.241649 0.240772 0.239904 0.239046 0.238197 0.237356 0.236525 ...], [0.2 0.199502 0.199007 0.198517 0.19803 0.197546 0.197066 0.196589 ...], ...]
```
<!-- pypto-doc-output:rsqrt:end -->
