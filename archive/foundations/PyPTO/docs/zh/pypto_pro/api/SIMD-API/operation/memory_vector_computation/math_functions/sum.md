# pypto_pro.language.sum

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

沿指定维度对`src` tile求和。`dim=0`沿最后一维对每行求和，输出`[行数, 1]`；`dim=1`沿第一维对每列求和，输出`[1, 列数]`。

## 函数原型

```python
pypto_pro.language.sum(out, src, tmp, *, dim=0)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 规约结果tile |
| `src` | 输入 | 源tile |
| `tmp` | 输入 | 兼容性临时tile |
| `dim` | 输入 | 归约方向 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 数据类型：与`src`一致<br>`dim=0`时shape为`[行数, 1]`，须设`layout=pl.DN`；`dim=1`时shape为`[1, 列数]` |
| `src` | 输入 | `dim=0`支持INT16、INT32、INT64、UINT64、FP16和FP32；`dim=1`支持INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、FP16、BF16和FP32<br>shape：二维Vec ND Tile |
| `tmp` | 输入 | API必传的Vec Tile兼容性参数；当前`dim=0`降低到`TROWSUM`时不读写该参数。`dim=1`降低到`TCOLSUM`，默认非binary路径不使用该参数 |
| `dim` | 输入 | `0`：沿最后一维做行向归约；`1`：沿第一维做列向归约。默认值为`0` |

## 流水类型

V（向量计算流水）。

## 调用示例

### dim = 0

下面是一个完整kernel：对64×128 FP32源tile做行向求和，输出`[64, 1]`。注意输出tile需设`layout=pl.DN`。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def row_sum_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32],
                   out: pl.Tensor[[64, 1], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_out = pl.TileType(shape=[64, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                         layout=pl.DN)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_tmp = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt_out, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.sum(cur_out, cur_a, cur_tmp)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:row_sum:start -->
```bash
输入数据a：[[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
输出数据out：[[1.008000e+03], [5.104000e+03], [9.200000e+03], [1.329600e+04], ...]
```
<!-- pypto-doc-output:row_sum:end -->

### dim = 1

下面的kernel对64×128 FP32源tile做列向求和，输出`[1, 128]`。

```python
@pl.jit(auto_mutex=True)
def col_sum_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32],
                   out: pl.Tensor[[1, 128], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_out = pl.TileType(shape=[1, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_tmp = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt_out, addrs=0x10000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.sum(cur_out, cur_a, cur_tmp, dim=1)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:col_sum:start -->
```bash
输入数据a：[[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
输出数据out：[[6.400000e+04 6.401600e+04 6.403200e+04 6.404800e+04 6.406400e+04 6.408000e+04 6.409600e+04 6.411200e+04 ...]]
```
<!-- pypto-doc-output:col_sum:end -->

### FP16精度说明

FP16归约会受到输入量化、有限精度累加及输出舍入的影响。输入规模较大或数值较大时，设备计算结果可能与高精度参考结果存在差异。归约指令采用的累加顺序也可能影响最终结果。

以下为`sum(..., dim=1)`使用FP16输入时的设备实测结果：

<!-- pypto-doc-output:col_reduce_sum:start -->
```bash
输入数据a：[[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
输出数据z：[[6.390400e+04 6.390400e+04 6.390400e+04 6.416000e+04 6.416000e+04 6.416000e+04 6.419200e+04 6.419200e+04 ...]]
```
<!-- pypto-doc-output:col_reduce_sum:end -->
