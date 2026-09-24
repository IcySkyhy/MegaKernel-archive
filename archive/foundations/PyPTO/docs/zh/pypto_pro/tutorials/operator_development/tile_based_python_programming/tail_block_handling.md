# 尾块处理

当GM Tensor的shape不能被Tile shape整除时，最后一行或最后一列Tile只包含部分有效数据，这类Tile称为**尾块**。本节介绍如何在PyPTO Pro中使用`valid_shape`和[`pl.set_validshape`](../../../api/SIMD-API/operation/memory_vector_computation/transpose_and_element_access/set_validshape.md)限定尾块的有效区域，以及何时需要`compact`、`pad`和`pl.fillpad`。

## 理解尾块

以二维Tensor为例，设Tensor shape为`[M, N]`，Tile shape为`[TILE_M, TILE_N]`，两个方向的Tile数量为：

```python
m_tiles = (M + TILE_M - 1) // TILE_M
n_tiles = (N + TILE_N - 1) // TILE_N
```

当`M % TILE_M != 0`时会产生尾行，当`N % TILE_N != 0`时会产生尾列；两者同时出现时，右下角为尾角。

![二维Tensor中的满块、尾行、尾列和尾角](../../figures/pro_tail_tile_grid.png)

对于第`i`行、第`j`列Tile，当前有效行列数可按下式计算：

```python
valid_rows = pl.min(M - i * TILE_M, TILE_M)
valid_cols = pl.min(N - j * TILE_N, TILE_N)
```

## shape与valid_shape

`TileType.shape`和`TileType.valid_shape`描述的对象不同：

| 参数 | 作用 |
| --- | --- |
| `shape` | Tile的物理规格，决定片上缓冲区大小和寻址边界 |
| `valid_shape` | 当前Tile中真正有效的行列范围 |

![Tile物理shape与逻辑valid_shape的关系](../../figures/pro_tail_shape_validshape.png)

对于每次运行时有效尺寸可能不同的尾块，建议在[`TileType`](../../../api/SIMD-API/basic_data_structures/TileType.md)中显式声明动态有效形状：

```python
tile_type = pl.TileType(
    shape=[64, 128],
    dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Vec,
    valid_shape=[-1, -1],
)
```

`valid_shape=[-1, -1]`表示两个维度都由运行时决定。如果只有一个维度动态，也可以使用如`[64, -1]`的声明。有效形状必须为正整数，且不能超过`shape`对应维度。

## 标准处理流程

尾块处理的关键顺序是：**先设置有效形状，再搬入和计算**。

![尾块的计算、有效形状设置、搬入、计算和搬出流程](../../figures/pro_tail_processing_flow.png)

```python
tile_a = a_group.next()
tile_b = b_group.next()
tile_c = c_group.next()

valid_rows = pl.min(M - i * TILE_M, TILE_M)
valid_cols = pl.min(N - j * TILE_N, TILE_N)

# 必须先设置，使随后的 load、计算和 store 使用同一有效区。
pl.set_validshape(tile_a, [valid_rows, valid_cols])
pl.set_validshape(tile_b, [valid_rows, valid_cols])
pl.set_validshape(tile_c, [valid_rows, valid_cols])

pl.load_tile(tile_a, a, [i, j])
pl.load_tile(tile_b, b, [i, j])
pl.add(tile_c, tile_a, tile_b)
pl.store_tile(c, tile_c, [i, j])
```

`pl.set_validshape`会更新Tile或TileGroup的当前有效范围。在上述顺序中：

- `load`/`load_tile`只从GM搬入有效区域，避免尾块越界读。
- 向量计算使用当前有效区域。
- `store`/`store_tile`只写回有效区域，避免越界写。

> `pl.set_validshape`应在`load`之前调用，用于约束GM搬入；在`load`之后调用仅影响后续操作。

### Tile与TileGroup

每个缓冲区的有效形状不同时，对`current()`或`next()`返回的Tile调用`pl.set_validshape`：

```python
tile = tile_group.next()
pl.set_validshape(tile, [valid_rows, valid_cols])
```

如果TileGroup中的所有缓冲区在整个Kernel期间都使用同一有效形状，可以对TileGroup统一设置：

```python
pl.set_validshape(tile_group, [valid_rows, valid_cols])
```

对逐块变化的尾块，应在每次获取Tile后设置当前块的有效形状。

## compact的作用

`compact`描述搬运、重排或矩阵计算路径对Tile片上物理排布的解释方式；`valid_shape`描述Tile的实际有效区域。调用`pl.set_validshape`时按对应数据路径配置`compact`。

| 值 | 含义 | 典型用途 |
| --- | --- | --- |
| `None`或`0` | 不启用紧凑模式 | 满块或对应API不需要紧凑布局的路径 |
| `1` | normal紧凑模式 | Mat→Left/Right、Acc搬出等需要按有效尺寸紧凑排列的分形或Cube路径 |
| `2` | RowPlusOne模式 | 明确要求额外一行物理空间的特定NZ路径 |

当前A5的Vec ND `load`/`store`使用`valid_shape`控制实际搬运行列，使用物理`shape`作为UB跨度，因此本节的逐元素Vec尾块不需要配置`compact`。当数据路径涉及Mat→Left/Right、Acc搬出等分形转换，需要按动态M/N调整片上排布时，配置`compact=1`。

`compact=2`适用于明确要求RowPlusOne布局的特定路径。

## 何时需要pad和fillpad

`valid_shape`只标记哪些元素有效，不会自动给无效区域写入数值。如果后续操作会读取整个物理Tile，需要使用`pad`指定安全填充值，并通过`pl.fillpad`完成填充。

| 计算语义 | 建议填充值 |
| --- | --- |
| 逐元素加、减、乘，且计算与写回都遵循`valid_shape` | 通常不需要填充 |
| 求和 | `pl.TilePad.zero` |
| 求最大值或softmax前的最大值归约 | `pl.TilePad.min` |
| 求最小值 | `pl.TilePad.max` |

下面示例将`src`的无效区域填为0：

```python
src_type = pl.TileType(
    shape=[64, 128],
    dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Vec,
    valid_shape=[-1, -1],
)
dst_type = pl.TileType(
    shape=[64, 128],
    dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Vec,
    pad=pl.TilePad.zero,
)

src = src_group.next()
dst = dst_group.next()
pl.set_validshape(src, [valid_rows, valid_cols])
pl.load(src, x, [row_offset, col_offset])
pl.fillpad(dst, src)
```

`pad`指定填充语义，`pl.fillpad`才是执行填充的操作。矩阵计算尾块通常使用`valid_shape`和`compact=1`将有效尺寸传递给L1/L0和`matmul`，不应笼统地将所有矩阵尾块都归类为需要`fillpad`；是否填充取决于具体数据路径和后续算子语义。

## 完整示例：二维加法的四类尾块

下面的Kernel支持动态二维shape。它使用`64 × 128`的物理Tile，通过逐块计算`valid_rows`和`valid_cols`，同时处理满块、尾行、尾列和尾角。

```python
import pypto_pro.language as pl
import torch

TILE_M = 64
TILE_N = 128


@pl.jit(auto_mutex=True)
def add_tail_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    tile_type = pl.TileType(
        shape=[TILE_M, TILE_N],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
    )
    a_group = pl.make_tile_group(
        type=tile_type, addrs=[0x0000, 0x4000], mutex_ids=[0, 1])
    b_group = pl.make_tile_group(
        type=tile_type, addrs=[0x8000, 0xC000], mutex_ids=[2, 3])
    c_group = pl.make_tile_group(
        type=tile_type, addrs=[0x10000, 0x14000], mutex_ids=[30, 31])

    with pl.section_vector():
        m = x.shape[0]
        n = x.shape[1]
        m_tiles = (m + TILE_M - 1) // TILE_M
        n_tiles = (n + TILE_N - 1) // TILE_N

        for i in pl.range(0, m_tiles, 1):
            for j in pl.range(0, n_tiles, 1):
                tile_a = a_group.next()
                tile_b = b_group.next()
                tile_c = c_group.next()

                valid_rows = pl.min(m - i * TILE_M, TILE_M)
                valid_cols = pl.min(n - j * TILE_N, TILE_N)
                pl.set_validshape(tile_a, [valid_rows, valid_cols])
                pl.set_validshape(tile_b, [valid_rows, valid_cols])
                pl.set_validshape(tile_c, [valid_rows, valid_cols])

                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])

device = "npu:0"
x = torch.randn(129, 257, dtype=torch.float16, device=device)
y = torch.randn_like(x)
z = torch.empty_like(x)
add_tail_kernel[None, 1](x, y, z)
torch.npu.synchronize()
torch.testing.assert_close(z, x + y, rtol=1e-3, atol=1e-3)
```

`129 × 257`在两个维度上均包含尾块，因此示例覆盖满块、行尾块、列尾块和角尾块四类Tile，并分别设置对应的`valid_shape`。

## 常见问题

### 在load之后设置valid_shape

```python
# 错误：本次 load 已经发生，无法再用 valid_shape 限制它。
pl.load(tile, x, offsets)
pl.set_validshape(tile, [valid_rows, valid_cols])
```

应调整为：

```python
pl.set_validshape(tile, [valid_rows, valid_cols])
pl.load(tile, x, offsets)
```

### 测试只使用满块shape

例如物理Tile为`[64, 128]`，测试仍使用`[64, 128]`的Tensor，只能证明满块路径可用，不能证明尾块不越界。尾块测试应至少包含一组两个维度均小于物理Tile的shape，或一组两个维度均不整除Tile的大shape。

### 只给输入Tile设置valid_shape

输入Tile、计算结果Tile和写回Tile应对同一逻辑区域使用一致的有效形状。遗漏输出Tile可能导致越界写或写回无效数据。

### 把pad当成自动填充

`pad`只声明填充值，不会单独产生填充操作。需要对无效区域进行实际填充时，调用`pl.fillpad`。

### 对所有尾块使用fillpad

逐元素计算通常只需要正确设置有效形状。只有后续操作会读取无效区域，且无效值会影响结果时，才需要选择与计算语义匹配的填充值。

## 参数选择速查

| 场景 | `valid_shape` | `compact` | `pad` / `fillpad` |
| --- | --- | --- | --- |
| 固定shape且全部为满块 | 默认或与`shape`一致 | 按对应API要求 | 不需要 |
| 向量逐元素ND动态尾块 | `[-1, -1]`，逐块调用`pl.set_validshape` | 不需要 | 通常不需要 |
| 尾块后执行求和 | `[-1, -1]` | 按对应API要求 | `zero` + `pl.fillpad` |
| 尾块后执行最大值归约 | `[-1, -1]` | 按对应API要求 | `min` + `pl.fillpad` |
| Cube动态尾块 | 为Mat/Left/Right/Acc设置对应有效尺寸 | `1` | 由具体数据路径和算子语义决定 |
