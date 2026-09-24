# pypto_pro.language.xor

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

逐元素按位异或：`out = lhs ^ rhs`。`rhs`须为与`lhs`同类型的Tile，计算需要临时Tile。

## 函数原型

```python
pypto_pro.language.xor(out, lhs, rhs, tmp)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 目标tile，存放按位异或结果 |
| `lhs` | 输入 | 左操作数tile |
| `rhs` | 输入 | 右操作数Tile |
| `tmp` | 输入 | 临时tile（中间计算用） |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 数据类型：8/16/32位整型<br>shape须与`lhs`、`rhs`一致 |
| `lhs` | 输入 | 数据类型：与`out`一致<br>shape：与`out`一致 |
| `rhs` | 输入 | 数据类型、row-major布局和有效shape与`out`一致 |
| `tmp` | 输入 | 数据类型：与`out`一致<br>shape：与`out`一致<br>硬件中间计算用，不可与`out`/`lhs`/`rhs`重叠 |

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整kernel：从GM载入两个INT32输入到UB，逐元素按位异或后写回GM。`tmp`为异或计算所需的临时tile。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def xor_kernel(
    a: pl.Tensor[[64, 64], pl.DT_INT32],
    b: pl.Tensor[[64, 64], pl.DT_INT32],
    out: pl.Tensor[[64, 64], pl.DT_INT32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_tmp = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    tile_out = pl.make_tile_group(type=tt, addrs=0xC000, mutex_ids=[3])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_tmp = tile_tmp.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.xor(cur_out, cur_a, cur_b, cur_tmp)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:xor:start -->
```bash
输入数据a：[[2 3 4 5 6 7 8 9 ...], [66 67 68 69 70 71 72 73 ...], [130 131 132 133 134 135 136 137 ...], [194 195 196 197 198 199 200 201 ...], ...]
输入数据b：[[1 2 3 4 5 6 7 8 ...], [1 2 3 4 5 6 7 8 ...], [1 2 3 4 5 6 7 8 ...], [1 2 3 4 5 6 7 8 ...], ...]
输出数据out：[[3 1 7 1 3 1 15 1 ...], [67 65 71 65 67 65 79 65 ...], [131 129 135 129 131 129 143 129 ...], [195 193 199 193 195 193 207 193 ...], ...]
```
<!-- pypto-doc-output:xor:end -->
