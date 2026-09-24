# pypto_pro.language.minimum

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

`minimum`同时支持逐元素取较小值和按维度取最小值归约。是否传入`dim`决定调用模式。

- **tile-tile模式**：`minimum(out, lhs, rhs)` -> `out[i] = min(lhs[i], rhs[i])`
- **tile-scalar模式**：`minimum(out, lhs, scalar)` -> `out[i] = min(lhs[i], scalar)`
- **归约模式**：`minimum(out, src, tmp, dim=0/1)`，`dim=0`沿最后一维归约，`dim=1`沿第一维归约

## 函数原型

```python
pypto_pro.language.minimum(out, lhs, rhs, *, dim=None)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 目标tile，存放逐元素结果或归约结果 |
| `lhs` | 输入 | 逐元素模式下为左操作数tile；归约模式下为源tile |
| `rhs` | 输入 | 逐元素模式下为右操作数（tile或scalar）；归约模式下为临时tile |
| `dim` | 输入 | `None`表示逐元素模式；`0`或`1`表示归约模式 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 逐元素模式支持8/16/32/64位整型、FP16、BF16和FP32，shape与`lhs`一致<br>归约模式：数据类型与`lhs`一致；`dim=0`时shape为`[行数, 1]`，`dim=1`时shape为`[1, 列数]` |
| `lhs` | 输入 | 逐元素模式：数据类型和shape均与`out`一致<br>归约模式：`dim=0`支持8/16/32位整型、FP16和FP32；`dim=1`还支持64位整型和BF16，shape为二维Tile |
| `rhs` | 输入 | tile-tile模式：数据类型与`out`一致，shape与`out`一致<br>tile-scalar模式：scalar值（int/float/Scalar）<br>归约模式：数据类型和shape均与`lhs`一致的临时tile |
| `dim` | 输入 | `None`：逐元素取较小值；`0`：沿最后一维取每行最小值；`1`：沿第一维取每列最小值。默认值为`None` |

## 流水类型

V（向量计算流水）。

## 调用示例

### tile-tile模式

下面是一个完整kernel：从GM载入两个FP32输入到UB，用`pypto_pro.language.minimum`逐元素取较小值后写回GM。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def minimum_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], b: pl.Tensor[[64, 64], pl.DT_FP32],
                   out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.minimum(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:minimum:start -->
```bash
输入数据a：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [17 17.25 17.5 17.75 18 18.25 18.5 18.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [49 49.25 49.5 49.75 50 50.25 50.5 50.75 ...], ...]
输入数据b：[[10 10.5 11 11.5 12 12.5 13 13.5 ...], [42 42.5 43 43.5 44 44.5 45 45.5 ...], [74 74.5 75 75.5 76 76.5 77 77.5 ...], [106 106.5 107 107.5 108 108.5 109 109.5 ...], ...]
输出数据out：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [17 17.25 17.5 17.75 18 18.25 18.5 18.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [49 49.25 49.5 49.75 50 50.25 50.5 50.75 ...], ...]
```
<!-- pypto-doc-output:minimum:end -->

### tile-scalar模式

```python
# tile 每个元素与 scalar 值取较小值
pl.minimum(out, lhs, 100.0)
```

### 归约模式

```python
pl.minimum(row_out, src, tmp, dim=0)  # row_out shape：[行数, 1]
pl.minimum(col_out, src, tmp, dim=1)  # col_out shape：[1, 列数]
```

#### dim = 0实测结果

```python
@pl.jit(auto_mutex=True)
def row_min_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32],
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
        pl.minimum(cur_out, cur_a, cur_tmp, dim=0)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:row_min:start -->
```bash
输入数据a：[[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
输出数据out：[[-8], [24], [56], [88], ...]
```
<!-- pypto-doc-output:row_min:end -->

#### dim = 1实测结果

```python
@pl.jit(auto_mutex=True)
def col_min_kernel(a: pl.Tensor[[64, 128], pl.DT_FP32],
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
        pl.minimum(cur_out, cur_a, cur_tmp, dim=1)
        pl.store(out, cur_out, [0, 0])
```

<!-- pypto-doc-output:col_min:start -->
```bash
输入数据a：[[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...], [24 24.25 24.5 24.75 25 25.25 25.5 25.75 ...], [56 56.25 56.5 56.75 57 57.25 57.5 57.75 ...], [88 88.25 88.5 88.75 89 89.25 89.5 89.75 ...], ...]
输出数据out：[[-8 -7.75 -7.5 -7.25 -7 -6.75 -6.5 -6.25 ...]]
```
<!-- pypto-doc-output:col_min:end -->
