# pypto_pro.language.range

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

在kernel中定义`for`循环范围，参数形式与Python `range`一致。`pl.range`是kernel内`for`循环唯一的合法迭代器，循环变量作为运行时标量参与索引和运算。

## 函数原型

```python
pypto_pro.language.range(stop)
pypto_pro.language.range(start, stop, step=1)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `start` | 输入 | 起始值（单参数形式下省略，默认`0`） |
| `stop` | 输入 | 终止值（不包含） |
| `step` | 输入 | 步长，默认`1` |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `start` / `stop` / `step` | 输入 | 整型常量、运行时整型标量或整型标量表达式（如`M // TILE_M`）<br>只支持1～3个位置参数，不支持关键字传参；`step`不能为0。正步长在迭代值达到或超过`stop`时结束，负步长在迭代值达到或小于`stop`时结束 |

## 调用示例

下面是一个完整kernel：用双层`pl.range`循环遍历`[M, N]`的GM Tensor，按`[TILE_M, TILE_N]`分块load两个输入，逐元素相加后store回GM。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl

TILE_M = 64
TILE_N = 64


@pl.jit(auto_mutex=True)
def for_add_fp16_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    M = x.shape[0]
    N = x.shape[1]
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x4000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x8000, mutex_ids=[30, 31])
    with pl.section_vector():
        for i in pl.range(0, M // TILE_M, 1):
            for j in pl.range(0, N // TILE_N, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
```

单参数形式等价于`range(0, stop, 1)`：

```python
for i in pl.range(10):
    ...
```
