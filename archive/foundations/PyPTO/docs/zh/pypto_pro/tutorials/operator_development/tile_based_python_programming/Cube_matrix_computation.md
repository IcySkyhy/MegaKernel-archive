# Cube矩阵计算编程

本文介绍如何在PyPTO Pro中使用Tile API编写基于L1 Buffer、L0A Buffer、L0B Buffer、L0C Buffer等片上存储的**矩阵计算代码**。

Cube计算单元专用于执行矩阵乘加运算，直接访问的专用缓存如下：L0A Buffer存储左矩阵，L0B Buffer存储右矩阵，L0C Buffer存储累加值及矩阵计算结果。PyPTO Pro通过`pl.section_cube()`标记Cube执行域，在域内完成数据搬运与矩阵计算。

## 矩阵编程的基本步骤

Cube矩阵计算的基本步骤为：数据搬入 → 数据加载 → 计算 → 数据搬出。当描述一个常见的Cube矩阵计算时，需要执行以下四个步骤：

1. 通过`pl.load`将数据从GM搬入L1 Buffer（Mat）
2. 通过`pl.move`将L1 Buffer数据搬入L0A Buffer（Left）和L0B Buffer（Right）
3. 通过`pl.matmul`执行矩阵乘法，结果存储在L0C Buffer（Acc）中
4. 通过`pl.store`将L0C Buffer中的结果搬出到GM

对应的数据流和硬件流水如下：

**图1**Cube矩阵计算的数据流和硬件流水

![Cube矩阵计算的数据流和硬件流水](../../figures/cube_matrix_computation_data_flow.png)

## 矩阵计算内存管理

### Cube侧Tile分配

Cube矩阵计算使用L0A/L0B/L0C Buffer，并经L1 Buffer中转。输入数据从GM搬入L1 Buffer（`pl.MemorySpace.Mat`），再从L1 Buffer搬入L0A（`pl.MemorySpace.Left`）/L0B（`pl.MemorySpace.Right`）；`pl.matmul`执行矩阵计算，并将结果写入L0C（`pl.MemorySpace.Acc`）。

PyPTO Pro通过`TileType`描述Tile的shape、dtype和target_memory。`TileType`本身不分配片上缓冲区，需要将其传给`pl.make_tile`或`pl.make_tile_group`。

本节代码仅展示Tile分配和同步方式，省略了完整Kernel的计算、调用及结果验证代码；整体代码结构及调用方式请参考本文末尾的[完整示例](#完整示例)。

各内存空间的典型角色如下：

| `pl.MemorySpace` | 物理缓冲区 | 典型角色 |
|:---|:---|:---|
| `Mat` | L1 Buffer | GM与L0A/L0B之间的矩阵暂存 |
| `Left` | L0A Buffer | `matmul`左操作数 |
| `Right` | L0B Buffer | `matmul`右操作数 |
| `Acc` | L0C Buffer | `matmul`累加结果（通常为FP32/INT32） |

#### 使用make_tile分配单个Tile

`pl.make_tile`分配一块固定的片上缓冲区。指定`addr`时需要同时指定`size`，其中`size`为缓冲区的字节数。使用`make_tile`时，跨Pipe依赖通过显式的`sync_src`/`sync_dst`对进行同步。

```python
TILE_M = 128
TILE_K = 128
TILE_N = 128

a_l1_type = pl.TileType(
    shape=[TILE_M, TILE_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
b_l1_type = pl.TileType(
    shape=[TILE_K, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)
a_left_type = pl.TileType(
    shape=[TILE_M, TILE_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left)
b_right_type = pl.TileType(
    shape=[TILE_K, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right)
acc_type = pl.TileType(
    shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc)

a_l1 = pl.make_tile(a_l1_type, addr=0x00000, size=32768)
b_l1 = pl.make_tile(b_l1_type, addr=0x08000, size=32768)
a_left = pl.make_tile(a_left_type, addr=0x0000, size=32768)
b_right = pl.make_tile(b_right_type, addr=0x0000, size=32768)
acc = pl.make_tile(acc_type, addr=0x0000, size=65536)
```

#### 使用make_tile_group分配轮转Tile

`pl.make_tile_group`分配一组轮转的Tile。`mutex_ids`的长度就是组内Tile数量，可通过`next()`、`current()`和`previous()`选择Tile。配合`@pl.jit(auto_mutex=True)`时，框架根据每个Tile的`mutex_id`自动插入跨Pipe同步。单缓冲也可以使用长度为1的`mutex_ids`，从而复用自动同步机制。

```python
@pl.jit(auto_mutex=True)
def matmul_kernel(a: pl.Tensor[[pl.DYNAMIC, TILE_K], pl.DT_FP16],
                  b: pl.Tensor[[TILE_K, pl.DYNAMIC], pl.DT_FP16],
                  out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32]):
    # L1使用双缓冲。
    a_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE_M, TILE_K], dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE_K, TILE_N], dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])

    # L0A、L0B和L0C使用单缓冲，并由auto_mutex管理同步。
    a_left = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE_M, TILE_K], dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[4])
    b_right = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE_K, TILE_N], dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[5])
    acc = pl.make_tile_group(
        type=pl.TileType(
            shape=[TILE_M, TILE_N], dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[6])
```

两种分配方式的区别如下：

| 方面 | `make_tile` | `make_tile_group` |
|:---|:---|:---|
| 缓冲区组织 | 单块固定缓冲区，使用`addr`和`size` | 一组轮转缓冲区，使用`addrs`和`mutex_ids` |
| 缓冲区选择 | 直接使用Tile变量 | 通过`next()`、`current()`、`previous()`选择 |
| 跨Pipe同步 | 手动插入`sync_src`/`sync_dst` | 带mutex元数据且配合`auto_mutex=True`时自动插入 |
| 适用场景 | 需要精确控制同步时序 | 单缓冲、双缓冲及N缓冲等常规场景 |

常规单缓冲、双缓冲及N缓冲场景使用`make_tile_group`并启用`auto_mutex=True`；需要精确控制同步事件及插入位置的场景使用`make_tile`和显式同步。

### 矩阵计算分形介绍

#### Ascend Cube分形布局

昇腾Cube计算单元以分形块（Fractal）作为基本计算和搬运单位。传统ND线性布局按行连续存放矩阵，读取二维计算块时需要从多段地址逐行收集数据。分形布局通过搬运流水重排数据，使计算块在物理内存中连续存放，从而减少寻址开销并提高数据吞吐率。

输入分形的一边固定为16，另一边为`32B / sizeof(dtype)`。常用数据类型的分形大小为：

- FP32：A矩阵为16×8，B矩阵为8×16。
- FP16/BF16：A、B矩阵均为16×16。
- FP8（E4M3FN/E5M2）：A矩阵为16×32，B矩阵为32×16，累加结果为FP32。
- HF8：A矩阵为16×32，B矩阵为32×16，累加结果为FP32。
- INT8：A矩阵为16×32，B矩阵为32×16。

L0C中的结果分形固定为16×16。以FP32/INT32累加结果为例，一个结果分形占用`16 × 16 × 4B = 1024B`，因此需要显式描述硬件分形大小的累加场景使用`fractal=1024`。

下图以FP16类型的40×56矩阵为例，展示`compact=0`时的标准分形布局：GM中的`pl.ND` Tensor通过`pl.load`搬入L1 Buffer的`Mat` Tile，并转换为`pl.NZ`布局。有效区为40×56，按16×16分形对齐后的寻址边界为48×64；分形之间按列优先排列，分形内部按行优先排列。图中白色区域为有效数据，灰色区域为无效区域，其值未必为0。`valid_shape`描述有效区域，`pad`/`pl.fillpad`决定是否以及如何填充无效区域；`compact=1`会按`valid_shape`紧凑解释片上布局，不使用图2所示的完整标准分形边界。

**图2** PyPTO Pro中`pl.load`完成ND（GM）到Nz（L1 Mat）的分形转换

![PyPTO Pro中pl.load完成ND（GM）到Nz（L1 Mat）的分形转换](../../figures/cube_matrix_nd_to_nz.png)

#### 分形格式的命名

矩阵分形格式采用“大Y小x”命名法：

- 大Y（Z/N）表示多个分形之间的排列顺序：Z为row major（行主序），N为column major（列主序）。
- 小x（z/n）表示一个分形内部的元素排列顺序：z为row major（行主序），n为column major（列主序）。

PyPTO Pro使用大写的`pl.NZ`、`pl.ZN`等枚举表示文档中的Nz、Zn格式。以二维矩阵为例，几种常用格式的含义如下：

- **ND**：通用线性布局，通常用于GM中的输入和输出Tensor。
- **Nz**：分形之间按列主序排列，分形内部按行主序排列。对shape为`[M, N]`的矩阵，补齐并拆分为`[M1, M0, N1, N0]`后，物理排列顺序为`[N1, M1, M0, N0]`。
- **Zn**：分形之间按行主序排列，分形内部按列主序排列。对shape为`[K, N]`的矩阵，补齐并拆分为`[K1, K0, N1, N0]`后，物理排列顺序为`[K1, N1, N0, K0]`。

对于矩阵乘法`C = A × B`，Ascend 950PR/Ascend 950DT要求左矩阵A使用Nz格式，右矩阵B使用Zn格式，结果矩阵C使用Nz格式。左矩阵按行取数、右矩阵按列取数时，相应元素均能从连续地址读取。

Ascend 950PR/Ascend 950DT的默认数据路径如下：

- GM中的ND数据搬入`Mat`（L1）时转换为`pl.NZ`。
- A矩阵从`Mat`搬入`Left`（L0A）后保持`pl.NZ`。
- B矩阵从`Mat`搬入`Right`（L0B）时转换为`pl.ZN`。
- `matmul`的结果在`Acc`（L0C）中按`pl.NZ`存放。

当`target_memory`确定后，`layout`和`fractal`通常随之确定，上述默认路径可省略这两个参数。转置搬入等特殊场景需要显式指定`layout`。`layout`描述Tile的物理排布，`TileType.shape`保持逻辑轴语义；例如B矩阵在L0B中仍使用`[K, N]`描述shape，物理布局为`pl.ZN`。

下图以FP16输入、FP32累加为例，展示`Left`、`Right`、`Acc` Tile与PyPTO Pro接口的对应关系。

**图3** PyPTO Pro矩阵乘法的Nz × Zn = Nz分形组合（FP16输入）

![PyPTO Pro矩阵乘法的Nz × Zn = Nz分形组合](../../figures/cube_matrix_fractal_formats_950.png)

### Cube侧同步

Cube矩阵计算的四个步骤分别对应MTE2、MTE1、M、FIX四条流水线。各流水线异步执行，当一条流水线生产的数据被另一条流水线消费时，需要插入同步以保证数据依赖。

| 流水线 | 含义 | 典型操作 |
|:---|:---|:---|
| MTE2 | GM→L1搬运 | `pl.load`/`pl.load_tile` |
| MTE1 | L1→L0A/L0B搬运 | `pl.move` |
| M | 矩阵计算 | `pl.matmul`/`pl.matmul_acc` |
| FIX | L0C→GM搬运 | `pl.store`/`pl.store_tile` |

使用`make_tile_group`并通过`@pl.jit(auto_mutex=True)`启用自动同步时，框架根据Tile的使用关系和`mutex_id`插入`mutex_lock`/`mutex_unlock`。

使用`make_tile`时，框架不会自动插入跨Pipe同步，需要在生产操作之后、消费操作之前插入配对的`pl.system.sync_src`和`pl.system.sync_dst`。下面展示一次完整矩阵计算中的前向数据依赖：

```python
with pl.section_cube():
    pl.load(a_l1, a, [0, 0])
    pl.load(b_l1, b, [0, 0])
    pl.system.sync_src(
        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
    pl.system.sync_dst(
        set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

    pl.move(a_left, a_l1)
    pl.move(b_right, b_l1)
    pl.system.sync_src(
        set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)
    pl.system.sync_dst(
        set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)

    pl.matmul(acc, a_left, b_right)
    pl.system.sync_src(
        set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=2)
    pl.system.sync_dst(
        set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=2)
    pl.store(out, acc, [0, 0])
```

`sync_src`由生产流水线SET flag，`sync_dst`由消费流水线WAIT flag；两者的`set_pipe`、`wait_pipe`和`event_id`必须一致。静态`event_id`取值范围为`[0, 7]`；动态整数Scalar的运行时数值也必须在该范围内。同一ID只能在上一次同步已经消费后复用。循环复用Tile时还需处理消费完成后才能覆盖缓冲区的反向依赖；常规流水化场景使用`make_tile_group`和自动同步。

> [!NOTE]说明
> 当`matmul`/`matmul_acc`使用了`phase`参数时，M流水与FIX流水之间的同步由硬件unit_flag完成，框架不会自动插入该段同步。

## 矩阵数据搬入

矩阵搬入分为两跳：GM → L1（`pl.load`）和L1 → L0A/L0B（`pl.move`）。

### Ascend 950PR/Ascend 950DT的L1 Buffer内存结构

Ascend 950PR/Ascend 950DT的L1 Buffer总容量为512KB，由16个32KB的Bank组成。每个Bank包含1024行，每行32B；16个Bank进一步组织为8个Bank Group（BG），每个BG包含Bank0和Bank1。单个Bank同一时刻最多执行一次读或一次写；同一BG内的两个Bank允许一个读、另一个写，但不支持同时读或同时写。L1地址的位域组织如下：

```text
L1_ADDR[18:0] = {BANK[18], BANK_DEPTH[17:8], BG[7:5], BANK_WIDTH[4:0]}
```

各字段在地址中的范围如下：

| 地址位 | 字段 | 含义 |
|:---|:---|:---|
| `[4:0]` | `BANK_WIDTH` | 一行内的字节偏移 |
| `[7:5]` | `BG` | 选择8个Bank Group之一 |
| `[17:8]` | `BANK_DEPTH` | 选择Bank中的1024行之一 |
| `[18]` | `BANK` | 选择当前BG内的Bank0或Bank1 |

例如，`0x00000`和`0x00100`的`BANK`与`BG`字段相同，只是`BANK_DEPTH`不同，
因此落在同一个Bank的不同数据行；`0x00000`和`0x40000`的`BG`相同、`BANK`
不同，因此落在同一BG的两个Bank。连续的32B行地址`0x00000`、`0x00020`、…、
`0x000E0`依次选择BG0～BG7，到`0x00100`后`BANK_DEPTH`加1并回到BG0。

在PyPTO Pro中，`make_tile_group`的`addrs`决定L1 Tile的起始地址。规划A、B矩阵以及双缓冲地址时，除了避免地址范围重叠，还应尽量避免并行访问落入存在冲突的Bank或Bank Group。

**图4** Ascend 950 L1 Buffer（`pl.MemorySpace.Mat`）内存结构

![Ascend 950 L1 Buffer（pl.MemorySpace.Mat）内存结构](../../figures/cube_matrix_l1_buffer_bank.png)

### GM → L1搬运

通过`pl.load`将矩阵从GM搬入L1 Buffer。`load`在搬运过程中自动完成ND到NZ的格式转换，无需手动配置分形参数。

```python
with pl.section_cube():
    cur_a = a_l1.current()
    cur_b = b_l1.current()
    pl.load(cur_a, a, [i, 0])    # A矩阵搬入L1，自动ND→NZ
    pl.load(cur_b, b, [0, j])    # B矩阵搬入L1，自动ND→NZ
```

`load`的坐标参数`[row, col]`为GM Tensor上的元素偏移，表示从该位置开始搬运一个Tile大小的数据。

### L1 → L0A/L0B搬运

通过`pl.move`将L1 Buffer中的数据搬入L0A/L0B Buffer，搬运过程中自动完成Nz到Zn（L0B）的格式转换。

```python
    cur_a_left = a_left.current()
    cur_b_right = b_right.current()
    pl.move(cur_a_left, cur_a)    # L1 → L0A，NZ→NZ
    pl.move(cur_b_right, cur_b)   # L1 → L0B，NZ→ZN
```

### 转置搬入

当输入矩阵的轴序与L1 Tile的轴序相反时（如Tensor为`[K, M]`而Tile为`[M, K]`），需要在搬入时进行转置。通过`load`的`order`参数控制：`order=[1, 0]`表示转置搬入，此时L1 Tile的`layout`需设为`pl.ZN`。

以`C[M, N] = A[M, K] @ B[K, N]`为例：

| 操作数 | Tensor shape | 是否转置 | `load`的`order` | L1 Mat Tile layout |
|:---|:---|:---|:---|:---|
| 左矩阵A | `[M, K]` | 否 | `[0, 1]`（默认） | `pl.NZ`（默认） |
| 左矩阵A | `[K, M]` | 是 | `[1, 0]` | `pl.ZN` |
| 右矩阵B | `[K, N]` | 否 | `[0, 1]`（默认） | `pl.NZ`（默认） |
| 右矩阵B | `[N, K]` | 是 | `[1, 0]` | `pl.ZN` |

左矩阵转置搬入示例：

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
```

## 矩阵数据搬出

通过`pl.store`将L0C Buffer中的计算结果搬出到GM。L0C→GM的搬运走FIX流水线，支持在搬运过程中进行格式转换（NZ→ND）。

```python
    pl.store(out, acc, [i, j])    # L0C → GM，自动NZ→ND
```

如果输出Tensor标注为NZ布局，`store`会将Acc的计算结果按NZ分形直接写入GM，无需额外格式转换：

```python
@pl.jit(auto_mutex=True)
def kernel(...,
           nz_out: pl.Tensor[[64, 64], pl.DT_FP32, pl.NZ],   # NZ输出Tensor
          ):
    ...
    pl.store(nz_out, acc, [0, 0])    # 按NZ分形写入GM
```

## 矩阵计算

`pl.matmul`是PyPTO Pro封装NPU硬件计算能力的矩阵乘法核心接口，实现`dst_tile = lhs_tile × rhs_tile`，数据通路为L0A(Left) × L0B(Right) → L0C(Acc)。

**表：矩阵乘计算A、B、C矩阵说明**

| 矩阵 | 存储位置 | 维度 | 数据格式 | 数据类型 |
|:---|:---|:---|:---|:---|
| A | L0A Buffer | M × K | Nz | FP16、BF16、FP32、INT8、HF8 |
| B | L0B Buffer | K × N | Zn | 与A一致 |
| C | L0C Buffer | M × N | Nz | FP16、BF16、FP32、INT32 |

```python
    pl.matmul(acc_tile, a_left, b_right)    # C = A × B
```

### MXFP8/MXFP4矩阵乘

MX矩阵乘使用`pl.matmul_mx`/`pl.matmul_mx_acc`，除Left/Right尾数Tile外，还需要分别位于L0A配套ScaleLeft缓冲区和L0B配套ScaleRight缓冲区的E8M0 scale Tile。每个scale对应K方向连续32个尾数元素，K必须为64的倍数。MXFP8支持E4M3/E5M2，MXFP4支持E2M1/E1M2；完整参数约束、scale Tensor布局和调用示例参见[`matmul_mx`](../../../api/SIMD-API/operation/matrix_computation/matmul_mx.md)和[`matmul_mx_acc`](../../../api/SIMD-API/operation/matrix_computation/matmul_mx_acc.md)。

### K维分块累加

当K维度较大，无法一次装入L1/L0时，需要将K轴切分为多个分块，逐块累加。首块用`pl.matmul`写入累加器，其余块用[`pl.matmul_acc`](../../../api/SIMD-API/operation/matrix_computation/matmul_acc.md)累加到同一个L0C。

K维分块累加对正确性有三个硬性要求：

1. **每步matmul / matmul_acc都要传`phase`**：首块和中间块用`phase=pl.AccPhase.Partial`，末块用`phase=pl.AccPhase.Final`；写回GM的`store`也传`phase=pl.STPhase.Final`。
2. **L0C累加器设`fractal=1024`**（FP32）。
3. **Cube段用`pl.system.set_mm_layout_transform(enabled=True)`开启**，段末`enabled=False`关闭。

```python
@pl.jit(auto_mutex=True)
def matmul_acc_kernel(
    a: pl.Tensor[[TILE, K_SIZE], pl.DT_FP16],
    b: pl.Tensor[[K_SIZE, TILE], pl.DT_FP16],
    c: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[4, 5])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[6, 7])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         fractal=1024),
        addrs=0x0000, mutex_ids=[8])

    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        ac = acc.current()
        for k in pl.range(0, K_SIZE, TILE):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_left.next()
            br = b_right.next()
            pl.load(cur_a, a, [0, k])
            pl.load(cur_b, b, [k, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if k == 0:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            elif k < K_SIZE - TILE:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(c, ac, [0, 0], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)
```

> [!NOTE]说明
> `phase`参数控制Cube（M流水）与FixPipe（FIX流水）之间的硬件unit_flag握手。`phase`配对使用时，框架不自动插入M与FIX之间的软件同步，由硬件unit_flag保证顺序。使用不当会导致精度问题或设备卡死。详见[`phase`使用约束](../../../api/SIMD-API/operation/matrix_computation/phase.md)。

## 尾块处理

当GM Tensor的shape不能被Tile shape整除时，边界上会出现比Tile小的尾块。Cube场景的尾块处理通过`valid_shape=[-1, -1]`配合`pl.set_validshape`和`compact=1`完成：

- `valid_shape=[-1, -1]`：声明有效区域为运行时动态，后续通过`set_validshape`设置。
- `compact=1`：让硬件按有效行紧凑摆放，减少L1/L0占用。

```python
tt_left = pl.TileType(shape=[TILE_M, TILE_K], dtype=pl.DT_FP16,
                      target_memory=pl.MemorySpace.Left,
                      valid_shape=[-1, -1], compact=1)
tt_right = pl.TileType(shape=[TILE_K, TILE_N], dtype=pl.DT_FP16,
                       target_memory=pl.MemorySpace.Right,
                       valid_shape=[-1, -1], compact=1)
tt_acc = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32,
                     target_memory=pl.MemorySpace.Acc,
                     valid_shape=[-1, -1], compact=1)
```

在运行时为每个尾块设置有效尺寸：

```python
valid_m = pl.min(TILE_M, M - i * TILE_M)
valid_n = pl.min(TILE_N, N - j * TILE_N)
pl.set_validshape(cur_acc, [valid_m, valid_n])
```

详细的尾块处理参数协同请参考[尾块处理](tail_block_handling.md)。

## 完整示例

以下是一个完整的Matmul Kernel，计算`C[M, N] = A[M, K] @ B[K, N]`，使用`make_tile_group` + `auto_mutex=True`管理L1/L0A/L0B/L0C缓冲，L1用双缓冲（`next()`轮转）让搬运与计算重叠：

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

TILE_M = 128
TILE_K = 128
TILE_N = 128


@pl.jit(auto_mutex=True)
def matmul_kernel(
    a: pl.Tensor[[pl.DYNAMIC, TILE_K], pl.DT_FP16],
    b: pl.Tensor[[TILE_K, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx()
    M = a.shape[0]
    N = b.shape[1]

    # L1双缓冲（next()轮转）
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, TILE_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_K, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])
    # L0A / L0B / Acc单缓冲（current()）
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, TILE_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[4])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_K, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[5])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[6])

    with pl.section_cube():
        for i in pl.range(core_id, M // TILE_M, num_cores):
            for j in pl.range(0, N // TILE_N, 1):
                cur_a = a_l1.next()
                cur_b = b_l1.next()
                pl.load_tile(cur_a, a, [i, 0])
                pl.load_tile(cur_b, b, [0, j])

                cur_a_left = a_left.current()
                cur_b_right = b_right.current()
                pl.move(cur_a_left, cur_a)
                pl.move(cur_b_right, cur_b)

                acc_tile = acc.current()
                pl.matmul(acc_tile, cur_a_left, cur_b_right)
                pl.store_tile(out, acc_tile, [i, j])


# Host端调用
device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
device = f"npu:{device_id}"
torch.npu.set_device(device)
torch.manual_seed(42)
M_SIZE, K_SIZE, N_SIZE = 8192, 128, 8192
a = torch.randn(M_SIZE, K_SIZE, device=device, dtype=torch.float16)
b = torch.randn(K_SIZE, N_SIZE, device=device, dtype=torch.float16)
out = torch.zeros(M_SIZE, N_SIZE, device=device, dtype=torch.float32)

matmul_kernel[None, 32](a, b, out)
torch.npu.synchronize()

golden = torch.matmul(a.float(), b.float())
torch.testing.assert_close(out, golden, rtol=1e-2, atol=1e-2)
print("Matmul kernel passed!")
```

> [!NOTE]说明
>
> - `make_tile_group`在`section_cube`外部声明，与Add等Vector示例风格一致。
> - L1使用双缓冲（`mutex_ids`长度为2），L0A/L0B/L0C使用单缓冲（`mutex_ids`长度为1）。
> - `auto_mutex=True`由框架自动管理各Tile的mutex锁。
> - 多核切分通过`pl.range(core_id, M // TILE_M, num_cores)`实现跨步分配，详见[多核切分与Tiling](multi_core_partitioning_and_Tiling.md)。
> - 上例K恰好为一个Tile，无需K维分块累加。K需要分块时请参考上文[K维分块累加](#k维分块累加)。
