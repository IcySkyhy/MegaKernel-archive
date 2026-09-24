# pypto_pro.language.TileType

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

描述一块Tile的“规格”——形状、数据类型、所在内存空间、排布方式等，配合[`pypto_pro.language.make_tile`](../operation/resource_management/make_tile.md)或[`pypto_pro.language.make_tile_group`](../operation/resource_management/make_tile_group.md)分配实际缓冲区。

TileType本身不分配内存，只是一个规格描述符。实际缓冲区通过`make_tile`（单块）或`make_tile_group`（多块轮转）创建。

## 函数原型

```python
pypto_pro.language.TileType(shape, dtype, target_memory=pypto_pro.language.MemorySpace.Vec, valid_shape=None,
             layout=None, fractal=None, pad=None, compact=None)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `shape` | 输入 | tile各维大小，如`[64, 128]` |
| `dtype` | 输入 | 元素数据类型，如`pypto_pro.language.DT_FP16` |
| `target_memory` | 输入 | 目标内存空间，默认`pypto_pro.language.MemorySpace.Vec` |
| `valid_shape` | 输入 | 可选，有效形状（处理尾块/非满块场景） |
| `layout` | 输入 | 可选，排布方式（`pl.NZ`/`pl.ZN`/`pl.ND`/`pl.DN`/`pl.ZZ`/`pl.NN`） |
| `fractal` | 输入 | 可选，分形大小 |
| `pad` | 输入 | 可选，填充模式 |
| `compact` | 输入 | 可选，紧凑模式 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `shape` | 输入 | 长度为2的编译期常量整数列表，各维大小须为正整数；仅支持二维Tile<br>对齐及分形布局约束由使用该TileType的具体API检查 |
| `dtype` | 输入 | [`pypto_pro.language.DataType`](DataType.md)枚举值<br>常用：`pypto_pro.language.DT_FP16`、`pypto_pro.language.DT_FP32`、`pypto_pro.language.DT_BF16`、`pypto_pro.language.DT_INT8`、`pypto_pro.language.DT_INT32` |
| `target_memory` | 输入 | [`pypto_pro.language.MemorySpace`](MemorySpace.md)枚举值<br>默认`pypto_pro.language.MemorySpace.Vec`（UB）<br>可选：`Vec`(UB)、`Mat`(L1)、`Left`(L0A)、`Right`(L0B)、`Acc`(L0C)、`Scaling`、`ScaleLeft`、`ScaleRight` |
| `valid_shape` | 输入 | 编译期常量整数列表或`None`（默认）<br>具体整数（如`[32, 64]`）：编译期确定有效形状<br>`None`：后端缺省行为等同于`[-1, -1]`（动态模式）<br>`[-1, -1]`：运行时动态设置有效形状，配合[`pypto_pro.language.set_validshape`](../operation/memory_vector_computation/transpose_and_element_access/set_validshape.md)使用 |
| `layout` | 输入 | [`pypto_pro.language.TensorLayout`](TensorLayout.md)枚举值或`None`（默认）<br>不指定时按“内存空间 + 架构”自动取默认值（见下表）<br>`Mat`、`Left`、`Right`、`Acc`、`Scaling`、`ScaleLeft`、`ScaleRight`的非法组合在构造时即报`ValueError`；`Vec`的可用布局由具体Tile API约束 |
| `fractal` | 输入 | 整数或`None`（默认）<br>`Acc`的FP32/INT32在未指定时自动设为1024<br>显式值会写入Tile硬件信息，取值要求和适用场景由具体Tile API决定 |
| `pad` | 输入 | [`pypto_pro.language.TilePad`](TilePad.md)枚举值或整数0-3<br>`null`(0)不填充 / `zero`(1)补零 / `max`(2)补最大值 / `min`(3)补最小值<br>非法值报`ValueError`，非法类型报`TypeError` |
| `compact` | 输入 | `None`（默认）或整数`0`、`1`、`2`<br>`None`/`0`：null；`1`：normal；`2`：row_plus_one<br>是否需要设置以及支持哪种模式，由使用该TileType的具体API和硬件路径决定。详见[`CompactMode`](CompactMode.md) |

## 默认布局表

`layout`不指定时，按内存空间和架构自动取默认值：

| 内存空间 | A3默认`layout` | A5默认`layout` |
|---|---|---|
| `Vec` | 无约束 | 无约束 |
| `Mat` | `pl.NZ` | `pl.NZ` |
| `Left` | `pl.ZZ` | `pl.NZ` |
| `Right` | `pl.ZN` | `pl.ZN` |
| `Acc` | `pl.NZ` | `pl.NZ` |
| `Scaling` | `pl.ND` | `pl.ND` |
| `ScaleLeft` | — | `pl.ZZ` |
| `ScaleRight` | — | `pl.NN` |

补充规则：

- `Mat`除默认`pl.NZ`外，还允许`pl.ZN`；当`dtype`为`UINT64`或`INT64`时，额外允许`pl.ND`
- `Left`跨架构兼容，同时允许`pl.ZZ`（A3默认）和`pl.NZ`（A5默认）
- `Mat`中的E8M0 scale允许`pl.ZZ`和`pl.NN`，未指定`fractal`时自动取32
- `ScaleLeft`和`ScaleRight`仅A5支持，分别用于A矩阵和B矩阵的E8M0 scale；未指定`fractal`时自动取32，Tile地址须按32字节对齐

## 调用示例

以下示例展示不同内存空间和使用场景下的`TileType`定义。`TileType`仅描述Tile规格，实际缓冲区由`make_tile`或`make_tile_group`创建。

```python
import pypto_pro.language as pl

TILE_M = 64
TILE_K = 128
TILE_N = 128

# UB Tile：target_memory 默认取 MemorySpace.Vec
vec_type = pl.TileType(
    shape=[TILE_M, TILE_N],
    dtype=pl.DT_FP16,
)

# A5 L0A Tile：Left 的默认布局为 NZ，也可以显式指定
left_type = pl.TileType(
    shape=[TILE_M, TILE_K],
    dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Left,
    layout=pl.NZ,
)

# A5 L1 转置分形 Tile：Mat 支持显式指定 ZN
mat_zn_type = pl.TileType(
    shape=[TILE_K, TILE_N],
    dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Mat,
    layout=pl.ZN,
)

# L0C Tile：FP32 Acc 未指定 fractal 时自动取 1024
acc_type = pl.TileType(
    shape=[TILE_M, TILE_N],
    dtype=pl.DT_FP32,
    target_memory=pl.MemorySpace.Acc,
)

# 动态尾块 Tile：运行时使用 set_validshape 设置实际有效形状
tail_type = pl.TileType(
    shape=[TILE_M, TILE_N],
    dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Vec,
    valid_shape=[-1, -1],
    compact=1,
)

# 带填充模式的 Tile：供 fillpad 等读取 pad 属性的操作使用
mask_type = pl.TileType(
    shape=[TILE_M, TILE_N],
    dtype=pl.DT_FP32,
    target_memory=pl.MemorySpace.Vec,
    valid_shape=[-1, -1],
    compact=1,
    pad=pl.TilePad.min,
)
```
