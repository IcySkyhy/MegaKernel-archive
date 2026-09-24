# CompactMode

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

Tile缓冲区的紧凑布局模式，通过[`pypto_pro.language.TileType`](TileType.md)的`compact`参数配置。调用时直接向`compact`传入`None`、`0`、`1`或`2`。

`compact`描述Tile在搬运、重排和矩阵计算路径中的布局解释方式，不改变Tile的数据类型，也不代替`valid_shape`对实际有效区域的描述。

## 取值

| 取值 | 整数 | 底层C++ 枚举 | 说明 | 典型用途 |
|---|---|---|---|---|
| `None`（默认）或`0`（null） | 0 | `CompactMode::Null` | 不启用紧凑模式 | 满块，或不需要紧凑布局的操作路径 |
| `1`（normal） | 1 | `CompactMode::Normal` | 按当前有效窗口使用普通紧凑模式 | 动态尾块参与`load`、`move`、`matmul`、Acc→Vec搬运或分形重排等路径 |
| `2`（row_plus_one） | 2 | `CompactMode::RowPlusOne` | 使用RowPlusOne紧凑模式 | 需要额外一行物理空间的NZ Tile搬运或`insert`路径，用于降低特定路径的bank conflict |

具体操作是否需要设置`compact`，以及支持哪一种模式，以对应API的参数约束为准。

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `compact` | 输入 | `None`、`0`、`1`或`2`。`None`与`0`最终均对应不启用紧凑模式；`1`对应normal；`2`对应row_plus_one |

## 补充说明

`shape`、`valid_shape`、`pad`、`compact`的职责不同：

| 参数 | 作用 |
|---|---|
| `shape` | 描述Tile的物理规格和默认寻址边界 |
| `valid_shape` | 描述Tile中当前有效的行列范围；配置为`[-1, -1]`时，可在运行时通过`set_validshape`更新 |
| `pad` | 描述无效区域的填充值，如补零、补最大值或补最小值 |
| `compact` | 描述搬运、重排或矩阵计算路径如何解释Tile的紧凑布局 |

- 需要使用normal紧凑模式的动态尾块，可同时配置`valid_shape=[-1, -1]`和`compact=1`，并在运行时调用`set_validshape`设置实际有效形状。
- `compact=2`仅用于明确要求RowPlusOne布局的路径，不是普通动态尾块的通用配置。
- `compact`可配置于`Vec`、`Mat`、`Left`、`Right`和`Acc` Tile；配置后是否能用于某个操作，取决于该操作支持的内存空间、布局和数据类型组合。
- `compact`不会自动填充无效区域。需要填充时，应同时配置`pad`，或使用对应的填充操作。

## 调用示例

### compact = 1动态行尾块

```python
import pypto_pro.language as pl

TILE_LARGE = 128


@pl.jit(auto_mutex=True)
def call_kernel_tail_row(
    a: pl.Tensor[[72, TILE_LARGE], pl.DT_FP16],
    b: pl.Tensor[[TILE_LARGE, TILE_LARGE], pl.DT_FP16],
    out: pl.Tensor[[72, TILE_LARGE], pl.DT_FP32],
):
    # compact=1：使动态行尾块在 load、move 和 matmul 路径中按有效窗口紧凑解释。
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_LARGE, TILE_LARGE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat,
                         valid_shape=[-1, -1], compact=1),
        addrs=0x00000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_LARGE, TILE_LARGE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat),
        addrs=0x20000, mutex_ids=[1])
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_LARGE, TILE_LARGE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Left,
                         valid_shape=[-1, -1], compact=1),
        addrs=0x0000, mutex_ids=[2])
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_LARGE, TILE_LARGE], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[3])
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_LARGE, TILE_LARGE], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc,
                         valid_shape=[-1, -1], compact=1),
        addrs=0x0000, mutex_ids=[4])

    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.set_validshape(cur_a, [72, TILE_LARGE])
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.set_validshape(al, [72, TILE_LARGE])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.set_validshape(ac, [72, TILE_LARGE])
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])
```

### compact = 2 RowPlusOne UB半块

```python
with pl.section_vector():
    p_f16_db = pl.make_tile_group(
        type=pl.TileType(shape=[TKV // 2 + 1, TS_HALF * 2], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=[VA2, VA2B], mutex_ids=[0, 1])
    # compact=2：P 半块使用 RowPlusOne（NZ+1）模式，降低后续 insert 路径的 bank conflict。
    p_f16_main_db = pl.make_tile_group(
        type=pl.TileType(shape=[TKV // 2 + 1, TS_HALF], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1], layout=pl.NZ, compact=2),
        addrs=[VA2, VA2B], mutex_ids=[0, 1])
    p_f16_back_db = pl.make_tile_group(
        type=pl.TileType(shape=[TKV // 2 + 1, TS_HALF], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1], layout=pl.NZ, compact=2),
        addrs=[VA2_DN, VA2B_DN], mutex_ids=[0, 1])
```
