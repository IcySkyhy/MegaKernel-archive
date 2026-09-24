# pypto_pro.language.expand_div

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

将指定维度的单元素tile广播到`src`的shape后执行逐元素除法。`dim=0`广播`[行数, 1]` tile；`dim=1`广播`[1, 列数]` tile。

## 函数原型

```python
pypto_pro.language.expand_div(out, src, scalar, *, dim=0)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 结果tile，与`src`同shape |
| `src` | 输入 | 源tile |
| `scalar` | 输入 | `[行数, 1]` tile，广播到每列做逐元素除法 |
| `dim` | 输入 | 展开方向 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 数据类型：与`src`一致<br>shape：与`src`一致<br>支持与`src`为同一tile，实现in-place |
| `src` | 输入 | 数据类型：8/16/32位整型、FP16、BF16和FP32<br>shape：`[行数, 列数]` |
| `scalar` | 输入 | 数据类型：与`src`一致<br>`dim=0`时shape为`[行数, 1]`，须设`layout=pl.DN`；`dim=1`时shape为`[1, 列数]`<br>元素值不能为0，否则硬件行为不确定 |
| `dim` | 输入 | `0`：广播`[行数, 1]` tile；`1`：广播`[1, 列数]` tile。默认值为`0` |

## 流水类型

V（向量计算流水）。

## 调用示例

### dim = 0

下面是一个完整kernel：把`[64, 1]`行向量广播到每列，与64×128源tile逐元素相除。注意行向量tile需设`layout=pl.DN`。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def row_expand_div_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32], v: pl.Tensor[[64, 1], pl.DT_FP32],
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
        pl.expand_div(cur_out, cur_a, cur_v)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:row_expand_div:start -->
```bash
输入数据a：[[-3 -2.875 -2.75 -2.625 -2.5 -2.375 -2.25 -2.125 ...], [13 13.125 13.25 13.375 13.5 13.625 13.75 13.875 ...], [29 29.125 29.25 29.375 29.5 29.625 29.75 29.875 ...], [45 45.125 45.25 45.375 45.5 45.625 45.75 45.875 ...], ...]
输入数据v：[[1.25], [1.5], [1.75], [2], ...]
输出数据out：[[-2.4 -2.3 -2.2 -2.1 -2 -1.9 -1.8 -1.7 ...], [8.666667 8.75 8.833333 8.916667 9 9.083333 9.166667 9.25 ...], [16.571428 16.642857 16.714285 16.785715 16.857143 16.928572 17 17.071428 ...], [22.5 22.5625 22.625 22.6875 22.75 22.8125 22.875 22.9375 ...], ...]
```
<!-- pypto-doc-output:row_expand_div:end -->

### dim = 1

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def col_expand_div_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32], v: pl.Tensor[[1, 128], pl.DT_FP32],
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
        pl.expand_div(cur_out, cur_a, cur_v, dim=1)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:col_expand_div:start -->
```bash
输入数据a：[[-3 -2.875 -2.75 -2.625 -2.5 -2.375 -2.25 -2.125 ...], [13 13.125 13.25 13.375 13.5 13.625 13.75 13.875 ...], [29 29.125 29.25 29.375 29.5 29.625 29.75 29.875 ...], [45 45.125 45.25 45.375 45.5 45.625 45.75 45.875 ...], ...]
输入数据v：[[1.25 1.5 1.75 2 2.25 2.5 2.75 3 ...]]
输出数据out：[[-2.4 -1.916667 -1.571429 -1.3125 -1.111111 -0.95 -0.818182 -0.708333 ...], [10.4 8.75 7.571428 6.6875 6 5.45 5 4.625 ...], [23.200001 19.416666 16.714285 14.6875 13.111111 11.85 10.818182 9.958333 ...], [36 30.083332 25.857143 22.6875 20.222221 18.25 16.636364 15.291666 ...], ...]
```
<!-- pypto-doc-output:col_expand_div:end -->
