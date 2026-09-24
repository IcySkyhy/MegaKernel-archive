# pypto_pro.language.store

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

把UB或L0C Tile的结果按**绝对元素坐标**写回GM，是与[`pypto_pro.language.load`](load.md)对应的写回接口。写出过程中可顺带融合ReLU、预量化，或对目的地址做原子累加。

源tile支持`Vec`(UB)和`Acc`(L0C)：源在`Acc`时由FIX（fixpipe）流水写回GM，源在`Vec`时走MTE3流水；不支持从`Mat`(L1)直接写回GM。

如果希望按"第几块tile"定位写出位置，需要使用[`pypto_pro.language.store_tile`](store_tile.md)。

## 函数原型

```python
pypto_pro.language.store(dst_tensor, src_tile, offsets, *, relu_pre_mode=None, scale=None,
         order=None, atomic=pl.AtomicType.AtomicNone, phase=None)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_tensor` | 输出 | 目标GM tensor，写出目的地 |
| `src_tile` | 输入 | 源tile，待写回GM的数据 |
| `offsets` | 输入 | GM写入的各维**绝对元素偏移**，长度等于`dst_tensor`维数 |
| `relu_pre_mode` | 输入 | 可选，写出前融合ReLU |
| `scale` | 输入 | 可选，随路量化比例（deqScalar / deqTensor路径） |
| `order` | 输入 | 可选，Tile维度在目标tensor维度中对应哪几根轴 |
| `atomic` | 输入 | 原子写模式 |
| `phase` | 输入 | 可选，matmul累加链路的fixpipe写回GM阶段 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_tensor` | 输出 | 数据类型：b8、b16、b32、b64<br>layout：支持`ND`、`DN`、`NZ`<br>offsets换算后的写入范围不得越过对应维度shape<br>原子累加时须预先初始化为零（或已知初值），累加在其上叠加 |
| `src_tile` | 输入 | 数据类型：b8、b16、b32、b64<br>内存空间：支持`Vec`(UB)或`Acc`(L0C)，不支持`Mat`(L1)；源在`Acc`时由FIX流水写回GM，源在`Vec`时走MTE3流水<br>首地址必须32字节对齐 |
| `offsets` | 输入 | 单位元素个数，大小不超过对应维度的shape，不支持负数索引 |
| `relu_pre_mode` | 输入 | 默认`None`（不融合ReLU）；可取`pl.ReluPreMode.NormalRelu`；与`scale`为`Tile`（per-channel）时互斥 |
| `scale` | 输入 | 可选，随路量化比例：`float`（编译期标量）→ per-tensor量化；运行时`FP32`标量→自动重解释为IEEE-754位模式；运行时`INT`标量→须传预编码的float32位模式（`struct.pack("!f", v)`）；`Tile`（INT64、`MemorySpace.Scaling`、shape `[1, N]`，`N % 16 == 0`且`N <= 512`）→ per-channel量化（`store_fp`路径），用户预制deqTensor tile，框架直接复用（不自动分配/同步，用户负责load→move→sync(MTE1→FIX)），与`relu_pre_mode`、`phase`互斥；`Tensor`不支持（per-channel须以`Tile`传入）；`[N, 1]`逐行量化不支持 |
| `order` | 输入 | 只支持配置tensor维度范围内的dim，只支持二维数组配置，其余配置报错<br>只支持升序排列（如 [0, 2]），不支持降序（如 [1, 0]）配置 |
| `atomic` | 输入 | `pl.AtomicType.AtomicNone`（默认，覆盖写）或`pl.AtomicType.AtomicAdd`（原子累加，硬件对每个目的地址做元素级加法） |
| `phase` | 输入 | `pl.STPhase.Unspecified`/`pl.STPhase.Partial`（中间累加步）/`pl.STPhase.Final`（最终步）；用硬件unit_flag接管cube/fixp间的握手，仅在matmul多步累加写回GM时使用；与`scale`为`Tile`（per-channel）时互斥 |

## 流水类型

MTE3（UB → GM的搬出流水）；当`src_tile`位于`Acc`(L0C)时为FIX（fixpipe）。

## 调用示例

下面是一个完整kernel：从GM载入两个输入，相加后用`pypto_pro.language.store`把UB上的结果写回GM。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def add_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16],
    b: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])

    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.add(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```

其他典型用法（节选）：

```python
# 写出前融合 ReLU
pl.store(relu_out, acc, [0, 0], relu_pre_mode=pl.ReluPreMode.NormalRelu)

# 原子累加（reduce 场景，多核/多步累加到同一 GM 位置）
pl.store(dk_out, fp32_row_tile, [b_id, g_id, single_indice, 0], atomic=pl.AtomicType.AtomicAdd)

# matmul 累加链路最终步写回 GM
pl.store(out_tensor, acc.current(), [i, j], phase=pl.STPhase.Final)
```
