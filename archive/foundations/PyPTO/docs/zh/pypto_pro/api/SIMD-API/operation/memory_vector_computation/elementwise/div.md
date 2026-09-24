# pypto_pro.language.div

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

两个操作数对应位置逐元素做除法。支持tile-tile和tile-scalar（scalar指标量）两种模式，支持in-place写法（`out`与`lhs`/`rhs`为同一tile）。

- **tile-tile模式**：`div(out, lhs, rhs)` -> `out[i] = lhs[i] / rhs[i]`
- **tile-scalar模式**：`div(out, lhs, scalar)` -> `out[i] = lhs[i] / scalar`

## 函数原型

```python
pypto_pro.language.div(out, lhs, rhs)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 目标tile，存放逐元素除法结果 |
| `lhs` | 输入 | 左操作数tile（被除数） |
| `rhs` | 输入 | 右操作数（tile或scalar，除数） |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 支持16/32/64位整型、FP16和FP32<br>shape须与`lhs`一致；支持与`lhs`或`rhs`为同一tile，实现in-place除法 |
| `lhs` | 输入 | 数据类型：与`out`一致<br>shape：与`out`一致 |
| `rhs` | 输入 | tile-tile模式：数据类型与`out`一致，shape与`out`一致<br>tile-scalar模式：scalar值（int/float/Scalar）<br>元素值不能为0，否则硬件行为不确定 |

## 流水类型

V（向量计算流水）。

## 调用示例

### tile-tile模式

下面是一个完整kernel：从GM载入两个FP32输入到UB，用`pypto_pro.language.div`逐元素相除后写回GM。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def div_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32], b: pl.Tensor[[64, 64], pl.DT_FP32],
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
        pl.div(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:div:start -->
```bash
输入数据a：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [17 17.25 17.5 17.75 18 18.25 18.5 18.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [49 49.25 49.5 49.75 50 50.25 50.5 50.75 ...], ...]
输入数据b：[[10 10.5 11 11.5 12 12.5 13 13.5 ...], [42 42.5 43 43.5 44 44.5 45 45.5 ...], [74 74.5 75 75.5 76 76.5 77 77.5 ...], [106 106.5 107 107.5 108 108.5 109 109.5 ...], ...]
输出数据out：[[0.1 0.119048 0.136364 0.152174 0.166667 0.18 0.192308 0.203704 ...], [0.404762 0.405882 0.406977 0.408046 0.409091 0.410112 0.411111 0.412088 ...], [0.445946 0.446309 0.446667 0.44702 0.447368 0.447712 0.448052 0.448387 ...], [0.462264 0.462441 0.462617 0.462791 0.462963 0.463134 0.463303 0.46347 ...], ...]
```
<!-- pypto-doc-output:div:end -->

### tile-scalar模式

```python
# tile 每个元素除以 scalar 值
pl.div(out, lhs, 2.0)
```
