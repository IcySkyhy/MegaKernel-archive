# Python语言编程概述

PyPTO Pro基于Python前端，提供了一套完整的DSL用于描述NPU上的Kernel计算。本章介绍PyPTO Pro编程中的核心数据结构与参数化机制。

## 核心数据结构

PyPTO Pro中有三种核心数据抽象：

| 抽象 | 所在位置 | 创建方式 | 用途 |
|:---|:---|:---|:---|
| **Tensor** | 全局内存（GM） | [`pl.Tensor[...]`](../../../api/SIMD-API/basic_data_structures/Tensor.md) / [`pl.make_tensor`](../../../api/SIMD-API/operation/resource_management/make_tensor.md) | 带shape和stride的GM Tensor视图 |
| **Tile** | 片上缓冲区 | [`pl.make_tile`](../../../api/SIMD-API/operation/resource_management/make_tile.md) | 固定的片上缓冲区（Vec / Mat / Left / Right / Acc / Scaling / ScaleLeft / ScaleRight） |
| **TileGroup** | 片上缓冲区 | [`pl.make_tile_group`](../../../api/SIMD-API/operation/resource_management/make_tile_group.md) | 一组轮转的Tile，用于双缓冲 / N缓冲 |

### Tensor

Tensor表示全局内存（GM）中带数据类型的视图，用作Kernel输入和输出。Kernel通过`load`将数据搬入片上Tile，并通过`store`将结果写回Tensor。

**在Kernel签名中声明**（最常见形式）：

```python
@pl.jit(auto_mutex=True)
def add_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],   # 输入GM Tensor
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],   # 输入GM Tensor
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],   # 输出GM Tensor
):
    ...
```

- `pl.Tensor[[shape...], dtype]` —— 第一个元素是shape列表，第二个是元素dtype
- `pl.DYNAMIC`：尺寸在启动时传入，尺寸变化不会产生新的编译变体
- `pl.STATIC`：尺寸在启动时读取并固化到IR，尺寸变化会选择新的编译变体
- 正整数：固定尺寸，启动时必须精确匹配
- Kernel内通过`x.shape[axis]`读取维度，支持负索引

**由裸指针构造运行时shape视图**：

```python
@pl.jit(auto_mutex=True)
def fa_kernel(
    q: pl.Ptr[pl.DT_FP16],
    k: pl.Ptr[pl.DT_FP16],
    v: pl.Ptr[pl.DT_FP16],
    o: pl.Ptr[pl.DT_FP16],
    tiling: OpTiling,
):
    # 由裸指针构造带类型的二维视图
    tensor_q = pl.make_tensor(q, [tiling.sq, tiling.d])
    ...
```

`pl.make_tensor(ptr, shape, stride=None, dtype=None)`由裸指针结合shape和可选stride构造Tensor视图。省略stride时，接口根据shape自动生成连续的行主序stride；需要表示非连续布局时，可继续传入显式stride。该接口是动态rank kernel的基础。

[`pl.make_ptr(tensor, dtype=None)`](../../../api/SIMD-API/operation/resource_management/make_ptr.md)从已有Tensor提取底层裸指针。
省略`dtype`时保留Tensor的元素类型；指定`dtype`时按目标元素类型解释指针，底层地址保持不变。

### Tile与TileType

Tile是一块固定的片上缓冲区。[`pl.TileType`](../../../api/SIMD-API/basic_data_structures/TileType.md)描述一个Tile的shape、dtype以及片上摆放，它本身不分配任何空间，而是传给`make_tile`/`make_tile_group`。

```python
@dataclass
class TileType:
    shape: Sequence[int]                 # 例如 [128, 128]
    dtype: DataType                      # pl.DT_FP16, pl.DT_FP32, ...
    target_memory: MemorySpace = MemorySpace.Vec
    valid_shape: Optional[Sequence[int]] = None  # 逻辑有效区域（< shape）
    layout: Optional[TensorLayout] = None        # Tile的分型（ND/DN/NZ/ZN/NN/ZZ）
    fractal: Optional[int] = None                # fractal大小
    pad: Optional[int] = None                    # TilePad.null/zero/max/min
    compact: Optional[int] = None                # 紧凑摆放模式
```

**内存空间**（`pl.MemorySpace`）：

| 空间 | 典型角色 |
|:---|:---|
| `Vec` | 向量单元工作缓冲区（UB）—— 逐元素计算 |
| `Mat` | L1 / 矩阵暂存缓冲区（GM → L1加载） |
| `Left` | L0A —— matmul左操作数 |
| `Right` | L0B —— matmul右操作数 |
| `Acc` | L0C —— matmul累加器（fp32/int32） |
| `Scaling` | 缩放/量化参数缓冲区 |
| `ScaleLeft` | L0A配套的MX左操作数E8M0 scale缓冲区 |
| `ScaleRight` | L0B配套的MX右操作数E8M0 scale缓冲区 |

> [!NOTE]说明
> 在同一款硬件上，**除了UB之外**，如果target_memory确定了，layout和fractal是确定的，可以不填。

### TileGroup

TileGroup是用`pl.make_tile_group`声明的一组轮转的Tile，用于实现双缓冲乃至更广义的N缓冲。当一块缓冲区正在被消费时，下一块可以同时被生产，从而让多条pipe重叠以提升吞吐。

```python
g = pl.make_tile_group(type=<TileType>, addrs=<base|list>, mutex_ids=[...], depth=<optional>)
```

- `type` —— 描述组中每一个Tile的`pl.TileType`
- `mutex_ids` —— 可选；每块Tile使用一个整数或非空整数列表/元组，ID取值范围`[0, 31]`。同一Tile内的ID不得重复，不同Tile可以复用ID
- `depth` —— Tile数量；`mutex_ids`为`None`或空列表时必填，非空时可由`len(mutex_ids)`推导
- `addrs` —— 单个基地址（Tile连续排布）或地址列表（每个Tile一个显式地址）

> [!NOTE]说明
> `mutex_ids=None`或`mutex_ids=[]`时，该group不参与`auto_mutex`，跨Pipe同步需由用户自行保证。

关于Tile和TileGroup的详细使用方法请参考[Tile矢量计算](Tile_vector_computation.md)和[Cube矩阵计算](Cube_matrix_computation.md)。

## 编程范式

PyPTO Pro的算子开发遵循「**搬入→计算→搬出**」三段式流水线范式，与AI Core硬件的多级异步流水特性完全贴合：

1. **搬入（CopyIn）**：通过[`pl.load`](../../../api/SIMD-API/operation/memory_data_movement/load.md)/[`pl.load_tile`](../../../api/SIMD-API/operation/memory_data_movement/load_tile.md)将数据从Global Memory搬运至片上缓冲区（UB/L1等）
2. **计算（Compute）**：在片上缓冲区上完成Tile级别的计算，根据算子类型在`pl.section_vector()`或`pl.section_cube()`上下文中调用对应的计算接口
3. **搬出（CopyOut）**：通过[`pl.store`](../../../api/SIMD-API/operation/memory_data_movement/store.md)/[`pl.store_tile`](../../../api/SIMD-API/operation/memory_data_movement/store_tile.md)将结果从片上缓冲区写回Global Memory

AI Core内部的搬运单元（MTE2/MTE1/MTE3等）与计算单元（V/M等）天然支持异步并行。通过TileGroup的N缓冲机制，可以让搬入下一块数据与当前块计算重叠执行，实现流水线吞吐叠加。

### 两档内存管理策略

PyPTO Pro提供两种Tile分配方式，对应不同的内存管理与同步复杂度：

| 策略 | 分配方式 | 同步管理 | 适用场景 |
|:---|:---|:---|:---|
| **自动同步（推荐）** | `pl.make_tile_group` + `auto_mutex=True` | 框架自动插入`mutex_lock`/`mutex_unlock` | 大多数Kernel；流水化/重叠的循环 |
| **手动同步** | `pl.make_tile` | 显式插入`sync_src`/`sync_dst`、barrier或`mutex_lock`/`mutex_unlock` | 需要精确放置同步操作的流水线 |

常规单缓冲、双缓冲及N缓冲场景使用`make_tile_group`并启用`auto_mutex=True`；需要精确控制同步事件及插入位置的场景使用`make_tile`和显式同步。两种方式可在同一Kernel中使用。

三类标准编程范式的详细实践请参考：

- [Tile矢量计算](Tile_vector_computation.md)：矢量类算子（Vector单元）
- [Cube矩阵计算](Cube_matrix_computation.md)：矩阵类算子（Cube单元）
- 融合类算子：通过stage机制联动Cube与Vector，框架自动完成Preload核间流水编排

## 同步机制概述

AI Core内部存在多条异步并行流水，当一条流水生产的数据被另一条流水消费时，必须插入同步事件确保数据依赖正确。PyPTO Pro提供两种同步模式：

### 自动同步（auto_mutex）

通过`@pl.jit(auto_mutex=True)`启用。框架根据TileGroup中每个Tile的`mutex_id`，在每次使用轮转Tile前后自动插入`mutex_lock`/`mutex_unlock`。该方式适用于常规单缓冲、双缓冲及N缓冲场景。

### 手动同步

跨Pipe生产者/消费者依赖可通过显式的`pl.system.sync_src`/`pl.system.sync_dst`对进行同步：

- `sync_src(set_pipe, wait_pipe, event_id)` —— 生产方SET flag
- `sync_dst(set_pipe, wait_pipe, event_id)` —— 消费方WAIT flag

需要显式控制缓冲区互斥时可使用[`mutex_lock`/`mutex_unlock`](../../../api/SIMD-API/operation/synchronization/mutex_lock_mutex_unlock.md)；需要等待指定pipe或本AI Core全部pipe上的前序操作完成时可使用[`bar_*`](../../../api/SIMD-API/operation/synchronization/barrier.md)。

PyPTO Pro的流水类型（`pl.PipeType`）与硬件指令流水对应关系：

| PipeType | 含义 | 典型操作 |
|:---|:---|:---|
| `MTE2` | GM→L1/UB搬运 | `pl.load`/`pl.load_tile` |
| `MTE1` | L1→L0A/L0B/ScaleLeft/ScaleRight搬运 | `pl.move` |
| `M` | 矩阵计算 | `pl.matmul` / `pl.matmul_mx` |
| `V` | 向量计算 | `pl.add`/`pl.sub`/... |
| `MTE3` | UB→GM搬运 | `pl.store`/`pl.store_tile` |
| `FIX` | L0C→GM搬运 | `pl.store`（Acc→GM） |

手动同步的典型模式为：搬入后插入`MTE2→V`同步确保数据就绪再计算，计算后插入`V→MTE3`同步确保计算完成再搬出。在循环场景下还需考虑反向同步（循环间依赖），防止当前迭代覆盖上一迭代未完成的数据。

> [!NOTE]说明
> `sync_src`/`sync_dst`的参数范围、配对及event ID复用要求参见[`sync_src`/`sync_dst`](../../../api/SIMD-API/operation/synchronization/sync_src_sync_dst.md)。手动同步属于ISASI类别的高级用法，不保证跨硬件版本兼容。

## TilingData

TilingData用于将**运行时参数**——shape、stride、循环边界、算子选择器、缩放系数等——传给已编译Kernel，而无需将它们固化在Kernel签名中。详细说明请参考[TilingData](TilingData.md)。
