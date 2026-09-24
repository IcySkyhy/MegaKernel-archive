# pypto_pro.language.make_tile

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

按[`TileType`](../../basic_data_structures/TileType.md)在指定内存空间的精确地址创建一个Tile，是kernel中创建Tile的核心接口。Tile的形状、数据类型、内存空间、排布等“规格”都由`TileType`描述，`make_tile`负责把它落到具体地址。

如果需要多块同规格tile做ping-pong双缓冲，并自动管理互斥，使用[`pypto_pro.language.make_tile_group`](make_tile_group.md)。

下图展示了`TileType`、`addr`和`size`如何共同确定Tile绑定的片上地址范围。

![make_tile创建Tile并绑定片上地址](../../../figures/make_tile_allocation.jpg "make_tile创建Tile并绑定片上地址")

## 函数原型

```python
pypto_pro.language.make_tile(tile_type, *, addr, size=None) -> tile
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `tile_type` | 输入 | `TileType`描述符，定义shape/dtype/内存空间/排布等；唯一的位置参数 |
| `addr` | 输入 | 必选关键字参数，Tile在该内存空间内绑定的起始地址（字节） |
| `size` | 输入 | 可选关键字参数，Tile绑定的地址范围大小（字节）；缺省时由`TileType`推导 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `tile_type` | 输入 | 须为[`pypto_pro.language.TileType`](../../basic_data_structures/TileType.md)，其内存空间决定该tile能用于哪些接口（如Left/Right用于matmul输入，Acc用于matmul输出） |
| `addr` | 输入 | 内存空间内的字节偏移，必选，且须为编译期常量<br>须以关键字传入（`addr=`）：与`size`同为两个裸整数时顺序无法自证，写反会把Tile放到错误地址<br>对齐要求：`Vec`（UB）和`Mat`（L1）为32字节，`Left`（L0A）和`Right`（L0B）为512字节，`Acc`（L0C）为64字节，`ScaleLeft`/`ScaleRight`为32字节<br>创建MX scale Tile时不能独立规划地址：`ScaleLeft` Tile地址必须等于配对`Left` Tile地址右移4位，`ScaleRight`与`Right`同理 |
| `size` | 输入 | 与`addr`对应的地址范围大小，单位为字节，须为编译期正整数<br>缺省时按“元素数 × dtype字节数”从`TileType`的shape和dtype推导，例如`[64, 128]`的FP16 Tile为16384字节，与[`make_tile_group`](make_tile_group.md)的每槽大小一致<br>仅当实际占用大于该值时才需显式指定，例如NZ/ZN排布按分形向上对齐后需要预留更多空间 |

## 调用示例

下面是一个完整kernel：用`pypto_pro.language.make_tile`将输入/输出Tile分别绑定到UB的三个地址区间，完成一次element-wise加法。纯vector kernel，同步用`sync_src`/`sync_dst`手写。

```python
import pypto_pro.language as pl


@pl.jit()
def make_tile_add_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    b: pl.Tensor[[64, 128], pl.DT_FP16],
    out: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    # size缺省，由tt推导为64 * 128 * 2 = 16384字节
    tile_a = pl.make_tile(tt, addr=0x0000)
    tile_b = pl.make_tile(tt, addr=0x4000)
    tile_out = pl.make_tile(tt, addr=0x8000)

    with pl.section_vector():
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(tile_out, tile_a, tile_b)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, tile_out, [0, 0])
```

> tile规格（shape/dtype/内存空间/排布/尾块）的完整说明见[`pypto_pro.language.TileType`](../../basic_data_structures/TileType.md)。
