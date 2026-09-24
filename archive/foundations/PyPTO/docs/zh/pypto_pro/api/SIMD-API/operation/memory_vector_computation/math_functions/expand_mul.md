# pypto_pro.language.expand_mul

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

将指定维度的单元素tile广播到`src`的shape后执行逐元素乘法。`dim=0`广播`[行数, 1]` tile；`dim=1`广播`[1, 列数]` tile。

## 函数原型

```python
pypto_pro.language.expand_mul(out, src, scalar, *, dim=0)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 结果tile，与`src`同shape |
| `src` | 输入 | 源tile |
| `scalar` | 输入 | `[行数, 1]` tile，广播到每列做逐元素乘法 |
| `dim` | 输入 | 展开方向 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 数据类型：与`src`一致<br>shape：与`src`一致<br>支持与`src`为同一tile，实现in-place |
| `src` | 输入 | 数据类型：8/16/32位整型、FP16、BF16和FP32<br>shape：`[行数, 列数]` |
| `scalar` | 输入 | 数据类型：与`src`一致<br>`dim=0`时shape为`[行数, 1]`，须设`layout=pl.DN`；`dim=1`时shape为`[1, 列数]` |
| `dim` | 输入 | `0`：广播`[行数, 1]` tile；`1`：广播`[1, 列数]` tile。默认值为`0` |

## 流水类型

V（向量计算流水）。

## 调用示例

### dim = 0

下面是一个完整kernel：把`[64, 1]`行向量广播到每列，与64×128源tile逐元素相乘。注意行向量tile需设`layout=pl.DN`。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def row_expand_mul_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32], v: pl.Tensor[[64, 1], pl.DT_FP32],
                          out: pl.Tensor[[64, 128], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_v = pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                       layout=pl.DN)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_v = pl.make_tile_group(type=tt_v, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_v = tile_v.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_v, v, [0, 0])
        pl.expand_mul(cur_out, cur_a, cur_v)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:row_expand_mul:start -->
```bash
输入数据a：[[-3 -2.875 -2.75 -2.625 -2.5 -2.375 -2.25 -2.125 ...], [13 13.125 13.25 13.375 13.5 13.625 13.75 13.875 ...], [29 29.125 29.25 29.375 29.5 29.625 29.75 29.875 ...], [45 45.125 45.25 45.375 45.5 45.625 45.75 45.875 ...], ...]
输入数据v：[[1.25], [1.5], [1.75], [2], ...]
输出数据out：[[-3.75 -3.59375 -3.4375 -3.28125 -3.125 -2.96875 -2.8125 -2.65625 ...], [19.5 19.6875 19.875 20.0625 20.25 20.4375 20.625 20.8125 ...], [50.75 50.96875 51.1875 51.40625 51.625 51.84375 52.0625 52.28125 ...], [90 90.25 90.5 90.75 91 91.25 91.5 91.75 ...], ...]
```
<!-- pypto-doc-output:row_expand_mul:end -->

### dim = 1

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def col_expand_mul_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32], v: pl.Tensor[[1, 128], pl.DT_FP32],
                          out: pl.Tensor[[64, 128], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_v = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_v = pl.make_tile_group(type=tt_v, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_v = tile_v.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_v, v, [0, 0])
        pl.expand_mul(cur_out, cur_a, cur_v, dim=1)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:col_expand_mul:start -->
```bash
输入数据a：[[-3 -2.875 -2.75 -2.625 -2.5 -2.375 -2.25 -2.125 ...], [13 13.125 13.25 13.375 13.5 13.625 13.75 13.875 ...], [29 29.125 29.25 29.375 29.5 29.625 29.75 29.875 ...], [45 45.125 45.25 45.375 45.5 45.625 45.75 45.875 ...], ...]
输入数据v：[[1.25 1.5 1.75 2 2.25 2.5 2.75 3 ...]]
输出数据out：[[-3.75 -4.3125 -4.8125 -5.25 -5.625 -5.9375 -6.1875 -6.375 ...], [16.25 19.6875 23.1875 26.75 30.375 34.0625 37.8125 41.625 ...], [36.25 43.6875 51.1875 58.75 66.375 74.0625 81.8125 89.625 ...], [56.25 67.6875 79.1875 90.75 102.375 114.0625 125.8125 137.625 ...], ...]
```
<!-- pypto-doc-output:col_expand_mul:end -->
