# pypto_pro.language.store_tile

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

把UB（`Vec`）或L0C（`Acc`）Tile的结果写回GM，与[`pypto_pro.language.store`](store.md)的区别在于：偏移以**tile块索引**为单位，内部自动按`块索引× tile_shape`换算成绝对元素坐标。是[`pypto_pro.language.load_tile`](load_tile.md)的反向操作。

例如tile shape为`[64, 128]`时，`tile_offsets=[2, 2]`等价于[`pypto_pro.language.store`](store.md)的绝对偏移`[128, 256]`。

下图以UB源Tile为例展示按块索引写回GM Tensor的过程。块索引先换算为元素偏移，再确定目标块的落点；L0C源Tile使用FIX流水。

![store_tile按块索引把Tile写回GM](../../../figures/store_tile_block_offset.jpg "store_tile按块索引把Tile写回GM")

## 函数原型

```python
pypto_pro.language.store_tile(dst_tensor, src_tile, tile_offsets, *, relu_pre_mode=None, scale=None, order=None, atomic=pl.AtomicType.AtomicNone, phase=None)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_tensor` | 输出 | 目标GM tensor，写出目的地 |
| `src_tile` | 输入 | 源Tile，内存空间只能是`Vec`(UB)或`Acc`(L0C) |
| `tile_offsets` | 输入 | 以tile为单位的块索引，内部换算为`块索引× tile_shape`的绝对元素偏移 |
| `relu_pre_mode` | 输入 | 可选，写回前融合ReLU |
| `scale` | 输入 | 可选，随路量化比例（deqScalar / deqTensor路径） |
| `order` | 输入 | 可选，Tile维度在目标tensor维度中对应哪几根轴 |
| `atomic` | 输入 | 可选，原子写模式，默认`pl.AtomicType.AtomicNone` |
| `phase` | 输入 | 可选，fixpipe分阶段写回模式 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_tensor` | 输出 | 数据类型：b8、b16、b32、b64<br>layout：支持`ND`、`DN`、`NZ`<br>换算后的写入范围不得越过对应维度shape |
| `src_tile` | 输入 | 数据类型：b8、b16、b32、b64<br>内存空间：只支持`Vec`(UB)和`Acc`(L0C)；UB源通过MTE3写回，L0C源通过FIX写回<br>地址对齐：UB为32字节，L0C为64字节 |
| `tile_offsets` | 输入 | 单位为tile块索引，支持运行时`Expr`，换算后的绝对偏移不超过对应维度的shape，不支持负数索引<br>被切分的维度（由`order`指定）按`块索引× tile该维大小`换算；其余维度的取值按绝对偏移直接使用 |
| `relu_pre_mode` | 输入 | 可选，支持`pl.ReluPreMode.NormalRelu`；与`scale`为`Tile`（per-channel）时互斥 |
| `scale` | 输入 | 可选，随路量化比例：`float`（编译期标量）→ per-tensor量化；运行时`FP32`标量→自动重解释为IEEE-754位模式；运行时`INT`标量→须传预编码的float32位模式（`struct.pack("!f", v)`）；`Tile`（INT64、`MemorySpace.Scaling`、shape `[1, N]`，`N % 16 == 0`且`N <= 512`）→ per-channel量化（`store_fp`路径），用户预制deqTensor tile，框架直接复用（不自动分配/同步，用户负责load→move→sync(MTE1→FIX)），与`relu_pre_mode`、`phase`互斥；`Tensor`不支持（per-channel须以`Tile`传入）；`[N, 1]`逐行量化不支持 |
| `order` | 输入 | 只支持配置tensor维度范围内的dim，只支持二维数组配置，其余配置报错<br>只支持升序排列（如 [0, 2]），不支持降序（如 [1, 0]）配置<br>用于高维tensor中指定tile对应哪几个维度；不配置时默认取tensor的最后两维 |
| `atomic` | 输入 | 支持`pl.AtomicType.AtomicNone`（覆盖写）或`pl.AtomicType.AtomicAdd`（原子累加） |
| `phase` | 输入 | 可选，支持`pl.STPhase.Partial`或`pl.STPhase.Final`；与`scale`为`Tile`（per-channel）时互斥 |

## 流水类型

源Tile位于`Vec`(UB)时为MTE3（UB → GM）；源Tile位于`Acc`(L0C)时为FIX（L0C → GM）。不支持从L1直接调用`store_tile`写回GM。

## 调用示例

下面是一个完整kernel：把同一块UB结果按块索引逐块写到GM输出的不同位置。`pypto_pro.language.store_tile`用块号`[ti, 0]`定位，内部自动换算为绝对偏移`[ti*64, 0]`。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def store_tile_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[256, 64], pl.DT_FP16],   # 4 个 64x64 的块
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_x = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])

    with pl.section_vector():
        cur_x = tile_x.current()
        pl.load(cur_x, x, [0, 0])
        for ti in pl.range(0, 4, 1):
            pl.store_tile(out, cur_x, [ti, 0])
```

其他典型用法（节选）：

```python
# 4D BSND tensor：tile对应第1、3维，其余维按绝对偏移
pl.store_tile(p_buf, p_f16, [b_idx, qi * 2 + sub_id, n_idx, ki], order=[1, 3])
```
