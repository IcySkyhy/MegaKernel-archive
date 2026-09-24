# pypto_pro.language.TensorLayout

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

数据布局枚举，用于描述GM Tensor的存储形式和Tile的数据排列形式。

- **GM Tensor**支持`ND`、`NZ`，默认`ND`。
- **Tile**支持`ND`、`DN`、`NZ`、`ZN`、`NN`、`ZZ`，默认值由内存空间和芯片架构决定。

## 取值

| 取值 | 数据排列 | 适用对象 | 典型用途 |
|---|---|---|---|
| `pl.ND` | 非分形行主序，最后一维连续 | Tensor / Tile | 普通GM Tensor（默认）；UB Tile（默认）；Scaling buffer |
| `pl.DN` | 非分形列主序 | Tile | UB上`[ROWS, 1]`列向量（归约结果、histogram索引） |
| `pl.NZ` | NZ分形排列 | Tensor / Tile | NZ GM Tensor（输出）；L1 Mat（默认）；A5的L0A（默认）；L0C Acc（默认） |
| `pl.ZN` | ZN分形排列 | Tile | L0B Right（默认）；转置搬入时的L1 Mat |
| `pl.ZZ` | ZZ分形排列 | Tile | A3的L0A（默认）；MX矩阵计算中，A矩阵的E8M0分组缩放因子在L1和ScaleLeft中的布局 |
| `pl.NN` | NN分形排列 | Tile | MX矩阵计算中，B矩阵的E8M0分组缩放因子在L1和ScaleRight中的布局 |

以上短名称分别等价于`pypto_pro.language.TensorLayout.ND`、`DN`、`NZ`、`ZN`、`NN`、`ZZ`。

---

## 布局声明

### Tensor布局

GM Tensor统一使用`ND`（行主序），`layout`可省略（默认`ND`）：

```python
x: pl.Tensor[[64, 128], pl.DT_FP16]                   # 默认 ND
x_nz: pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ]         # NZ 分形布局（用于输出）
```

MX矩阵计算使用的E8M0分组缩放因子在GM中仍声明为普通`ND` Tensor；物理shape和搬运约束见[`matmul_mx`](../operation/matrix_computation/matmul_mx.md)和[`load`](../operation/memory_data_movement/load.md)。

> [!IMPORTANT]重要
> 转置搬运由[`load`](../operation/memory_data_movement/load.md)/[`load_tile`](../operation/memory_data_movement/load_tile.md)的`order`参数决定（`order=[1,0]`即`is_transpose=True`），需与L1 Tile布局`ZN`配合。详见下文[转置搬入](#转置搬入)。

### Tile布局

Tile通过[`pl.TileType`](TileType.md)的`layout`参数指定。不指定时，默认值由内存空间和芯片架构决定：

| 内存空间 | A3默认 | A5默认 | 额外允许 |
|---|---|---|---|
| `Vec`（UB） | 无默认值 | 无默认值 | `ND`；`DN`（仅特定API要求的列主序场景） |
| `Mat`（L1） | `NZ` | `NZ` | `ZN`（转置搬入）；`DT_FP8E8M0`还允许`ZZ`、`NN`；`UINT64`/`INT64`还允许`ND` |
| `Left`（L0A） | `ZZ` | `NZ` | `ZZ`、`NZ` |
| `Right`（L0B） | `ZN` | `ZN` | — |
| `Acc`（L0C） | `NZ` | `NZ` | — |
| `Scaling` | `ND` | `ND` | — |
| `ScaleLeft` | — | `ZZ` | — |
| `ScaleRight` | — | `NN` | — |

---

## 调用示例

### 转置搬入

GM Tensor统一使用`ND`布局。Cube场景下，是否需要转置搬入由Tensor的shape决定：当传入的shape与L1 Tile的shape轴序一致时不需要转置；当轴序相反时需要转置，此时`load`设置`order=[1, 0]`（`is_transpose=True`），L1 Tile布局配`ZN`。`order`参数由框架内部解析为`is_transpose`标志（`order[0] > order[1]`即为转置），供codegen生成对应的TLOAD指令。

以`C[M, N] = A[M, K] @ B[K, N]`为例：

| 操作数 | Tensor shape | 是否转置 | `load`的`order` | L1 Mat Tile layout |
|---|---|---|---|---|
| 左矩阵A | `[M, K]` | 否 | `[0, 1]`（默认） | `NZ`（默认） |
| 左矩阵A | `[K, M]` | 是 | `[1, 0]` | `ZN` |
| 右矩阵B | `[K, N]` | 否 | `[0, 1]`（默认） | `NZ`（默认） |
| 右矩阵B | `[N, K]` | 是 | `[1, 0]` | `ZN` |

#### 左矩阵转置搬入

Tensor shape为`[K, M]`（与L1 Tile的`[M, K]`轴序相反），`load`设置`order=[1, 0]`，L1 Tile配`ZN`：

```python
@pl.jit(auto_mutex=True)
def kernel_left_transpose(
    a: pl.Tensor[[K, M], pl.DT_FP16],               # [K, M]，需转置
    b: pl.Tensor[[K, N], pl.DT_FP16],               # [K, N]，不转置
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN),  # ZN
        addrs=0x00000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat),                # NZ（默认）
        addrs=0x10000, mutex_ids=[1])
    ...
    with pl.section_cube():
        cur_a = a_l1.current()
        pl.load(cur_a, a, [0, 0], order=[1, 0])    # 转置搬入
        cur_b = b_l1.current()
        pl.load(cur_b, b, [0, 0])                   # 不转置
        ...
```

#### 右矩阵转置搬入

Tensor shape为`[N, K]`（与L1 Tile的`[K, N]`轴序相反），`load`设置`order=[1, 0]`，L1 Tile配`ZN`：

```python
@pl.jit(auto_mutex=True)
def kernel_right_transpose(
    a: pl.Tensor[[M, K], pl.DT_FP16],               # [M, K]，不转置
    b: pl.Tensor[[N, K], pl.DT_FP16],               # [N, K]，需转置
    out: pl.Tensor[[M, N], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[M, K], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat),                # NZ（默认）
        addrs=0x00000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[K, N], dtype=pl.DT_FP16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN),  # ZN
        addrs=0x10000, mutex_ids=[1])
    ...
    with pl.section_cube():
        cur_a = a_l1.current()
        pl.load(cur_a, a, [0, 0])                   # 不转置
        cur_b = b_l1.current()
        pl.load(cur_b, b, [0, 0], order=[1, 0])    # 转置搬入
        ...
```

### UB Tile的ND与DN

UB Tile大部分情况使用`ND`（行主序）。`DN`（列主序）仅在特定API要求时使用，典型场景是归约操作产生`[ROWS, 1]`列向量：

```python
# 普通数据 Tile：ND（行主序）
tile_src = pl.TileType(shape=[32, 128], dtype=pl.DT_UINT16,
                       target_memory=pl.MemorySpace.Vec, layout=pl.ND)

# 归约结果列向量：DN（列主序）
tile_red = pl.TileType(shape=[TILE_ROWS, 1], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Vec, layout=pl.DN)
```

### Cube分形布局

Cube计算的L1/L0A/L0B/L0C各级Buffer使用分形布局，默认值由内存空间决定：

```python
# L1 Mat：默认 NZ
mat_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Mat, layout=pl.NZ)

# L0B Right：默认 ZN
right_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN)

# L0C Acc：默认 NZ，fp32 需指定 fractal=1024
acc_type = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32,
                       target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024)
```

### A3架构L0A的ZZ布局

A3架构下L0A（左矩阵）默认`ZZ`；A5架构下默认`NZ`。以下为显式指定`ZZ`的用法：

```python
tile_type = pl.TileType(
    shape=[128, 128], dtype=pl.DT_FP16,
    target_memory=pl.MemorySpace.Left, layout=pl.ZZ,
)
```

### NZ Tensor输出

输出Tensor标注为`NZ`时，`pl.store`会将Acc的计算结果按NZ分形直接写入GM Tensor，无需额外格式转换：

```python
@pl.jit()
def store_nz_cce_kernel(
    ...
    nz_out: pl.Tensor[[64, 64], pl.DT_FP32, pl.NZ],   # NZ 输出 Tensor
    ...
):
    ...
    pl.store(nz_out, acc, [0, 0])    # 按 NZ 分形写入 GM
```

### MX scale的ZZ与NN布局

`ZZ`和`NN`分别用于存放MX矩阵计算中A矩阵和B矩阵的E8M0 scale。L1 Mat Tile默认使用`NZ`，因此A矩阵的scale需要显式指定`ZZ`，B矩阵的scale需要显式指定`NN`；ScaleLeft和ScaleRight Tile则分别默认使用`ZZ`和`NN`。

```python
# A/B矩阵的E8M0 scale逻辑shape分别为[M,G]和[G,N]，其中G=K/32。
M, G, N = 64, 4, 64

# L1 Mat的默认布局是NZ，因此需要显式指定ZZ或NN。
scale_a_l1_type = pl.TileType(
    shape=[M, G],
    dtype=pl.DT_FP8E8M0,
    target_memory=pl.MemorySpace.Mat,
    layout=pl.ZZ,
)
scale_b_l1_type = pl.TileType(
    shape=[G, N],
    dtype=pl.DT_FP8E8M0,
    target_memory=pl.MemorySpace.Mat,
    layout=pl.NN,
)

# ScaleLeft/ScaleRight分别默认使用ZZ/NN，无需再次指定layout。
scale_a_type = pl.TileType(
    shape=[M, G],
    dtype=pl.DT_FP8E8M0,
    target_memory=pl.MemorySpace.ScaleLeft,
)
scale_b_type = pl.TileType(
    shape=[G, N],
    dtype=pl.DT_FP8E8M0,
    target_memory=pl.MemorySpace.ScaleRight,
)
```
