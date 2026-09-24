# 多核切分与Tiling：把数据分到多个逻辑AI Core

本节介绍PyPTO Pro的多核切分方法，包括：

- 获取当前执行域的全局逻辑核索引和逻辑Block数；
- 在启动时设置逻辑Block数（`block_dim`）；
- 通过动态shape、标量参数或TilingData传递运行时Tiling数据；
- 根据任务量和硬件资源进行负载均衡切分。

配套阅读：[TilingData](TilingData.md)、[尾块Tile](tail_block_handling.md)、
[Tensor、Tile与TileGroup](Python_programming_overview.md)。

示例代码使用以下导入：

```python
import torch
import pypto_pro.language as pl
```

---

## 逻辑Block与执行域

`block_dim`是启动时配置的逻辑Block数，[`pl.get_block_num()`](../../../api/SIMD-API/operation/system_variables/get_block_num.md)返回本次启动传入的该值。仅启动Cube或仅启动Vector时，执行域中的逻辑核数与`block_dim`一致；同时启动AIC与AIV时，各执行域的逻辑核数由`block_dim`和AIC:AIV比例共同决定。

四个索引接口的语义如下：

- `pl.get_block_num()`：启动时配置的逻辑Block数；
- [`pl.get_block_idx()`](../../../api/SIMD-API/operation/system_variables/get_block_idx.md)：当前执行域的全局逻辑核索引；在Vector段中已按subblock展平；
- [`pl.get_subblock_idx()`](../../../api/SIMD-API/operation/system_variables/get_subblock_idx.md)：当前逻辑Block内的subblock索引，仅在需要区分同一Block内的AIV时使用；
- `pl.get_subblock_num()`：当前执行域每个逻辑Block对应的subblock数量。

仅包含一个执行域的Kernel可直接写成：

```python
num_cores = pl.get_block_num()     # 例如32
core_id = pl.get_block_idx()       # 0..31
```

当前支持的典型取值关系如下：

| Kernel执行模式 | 执行域 | `get_block_idx()`范围 | 该域工作单元总数 |
|---|---|---|---|
| 仅Cube或仅Vector | 当前执行域 | `[0, block_num)` | `block_num` |
| AIC:AIV为1:1的混合Kernel | Cube或Vector | `[0, block_num)` | `block_num` |
| AIC:AIV为1:2的混合Kernel | Cube | `[0, block_num)` | `block_num` |
| AIC:AIV为1:2的混合Kernel | Vector | `[0, 2 * block_num)` | `block_num * get_subblock_num()` |

其中`block_num = pl.get_block_num()`。在1:2混合Kernel的Vector段中，`pl.get_subblock_num()`返回2。

![1:2混合Kernel中逻辑Block与Cube、Vector执行域的索引映射](../../figures/pro_multicore_spmd_mapping.png)

上图以`block_dim=8`为例：Cube执行域包含8个AIC工作单元，Vector执行域包含16个AIV工作单元；Vector侧的`pl.get_block_idx()`是展平后的全局AIV逻辑索引。

### `get_subblock_idx`：逻辑Block内的subblock索引

在1:2混合Kernel中，`pl.get_subblock_idx()`可区分同一逻辑Block对应的两个AIV，返回`0`或`1`。例如让两个AIV分别处理Tile的前半行和后半行：

```python
with pl.section_vector():
    sub_id  = pl.get_subblock_idx()      # 0 或 1
    row_off = sub_id * TS_HALF           # 两个子核各处理 [0:TS_HALF] 和 [TS_HALF:TS]
```

Vector段的`pl.get_block_idx()`是包含subblock信息的全局AIV逻辑索引。混合Kernel的Vector段按全部AIV做跨步切分时，使用：

```python
with pl.section_vector():
    worker_id = pl.get_block_idx()
    worker_num = pl.get_block_num() * pl.get_subblock_num()
```

---

## 使用跨步循环进行多核切分

将总任务切分为`total_tiles`个Tile后，可以采用跨步（strided）方式分配任务：
第`core_id`个逻辑核处理编号为`core_id, core_id+num_cores, core_id+2*num_cores, ...`
的Tile。该方式可通过`pl.range(start, stop, step)`实现，其参数语义与Python `range`一致：

```python
with pl.section_vector():
    # 本例是仅包含Vector段的Kernel；混合Kernel请按上一节计算工作单元总数。
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx()

    total_tiles = m_tiles * n_tiles          # 整份工作被切成这么多 Tile

    # 每个核跨步遍历扁平化的 Tile 网格。
    for idx in pl.range(core_id, total_tiles, num_cores):
        i = idx // n_tiles                   # 行 Tile 索引
        j = idx % n_tiles                    # 列 Tile 索引
        ...
```

与连续分段相比，跨步切分具有以下特点：

- **负载均衡**：若`total_tiles`不能被`num_cores`整除，前
  `total_tiles % num_cores`个逻辑核各多处理1个Tile，各核任务量最多相差1。
- **循环边界统一**：所有核共用同一个`pl.range(core_id, total_tiles, num_cores)`，代码
  无需为首核或尾核单独设置分支。

也可以采用二维切分：外层使用`range(core_id, m_tiles, num_cores)`切分行，内层使用
`range(0, n_tiles, 1)`遍历列。应根据数据连续性和负载均衡需求选择切分方式。

![PyPTO Pro使用pl.range进行跨步多核切分](../../figures/pro_multicore_strided_partition.png)

上图用18个Tile和4个逻辑核展示跨步分配。每个核从自己的`core_id`开始，以
`num_cores`为步长继续认领Tile，因此各核工作量最多相差1。

### 跨步切分示例

设`total_tiles = 100`、`num_cores = 32`。`100 = 3 * 32 + 4`，跨步分配的结果是：

- 核`0..3`（共4个）各处理 **4** 个Tile，例如核0处理`{0, 32, 64, 96}`；
- 核`4..31`（共28个）各处理 **3** 个Tile，例如核4处理`{4, 36, 68}`。

各逻辑核的任务量最多相差1个Tile，且无需增加首核或尾核分支。若使用固定长度为4的
连续分段`[k*4 : k*4+4]`，则前25个核各处理4个Tile，后7个核不处理任务，负载分布不均衡。

### 二维切分与扁平切分

| 方式        | 循环写法                                                   | 适用                         |
|-------------|------------------------------------------------------------|------------------------------|
| **扁平切** | `for idx in pl.range(core_id, m_tiles*n_tiles, num_cores)` | Tile总数多、想要最细粒度均衡 |
| **二维切** | 外`range(core_id, m_tiles, num_cores)` + 内`range(0, n_tiles, 1)` | 每核处理整行、数据更连续 |

扁平切把`idx`还原成`(i, j)`：`i = idx // n_tiles`、`j = idx % n_tiles`。二维切让同一
核处理连续的若干整行，GM访问更连续，但当`m_tiles < num_cores`时会有核空转（此时应改
扁平切）。

![PyPTO Pro扁平切分与二维按行切分对比](../../figures/pro_multicore_flat_vs_2d.png)

图中的颜色表示Tile所属的AI Core。扁平切分追求更细粒度的负载均衡；二维按行切分让同核
访问更连续的数据。两种方式采用相同的SPMD编程模型，仅任务展开方式不同。

---

## 在启动时设置逻辑Block数（block_dim）

编译后的Kernel使用方括号语法启动：`kernel[stream, block_dim](*args)`。

```python
kernel[stream, block_dim](x, y, z)   # 显式Stream + 逻辑Block数
kernel[block_dim](x, y, z)           # 默认Stream，只给逻辑Block数
kernel(x, y, z)                       # 默认Stream，block_dim=1
```

- `stream`：NPU Stream；传`None`时使用默认Stream。
- `block_dim`：本次启动传入的逻辑Block数；Kernel中的`get_block_num()`返回该值。

```python
# 例：请求32个逻辑Block
kernel[None, 32](x, y, z)
torch.npu.synchronize()
```

### 根据硬件与任务量设置block_dim

`block_dim`应同时满足执行域硬件上限和任务并行度要求，可采用以下两种计算方式：

**(a) 查询平台核数**：

```python
from pypto_pro.runtime.platform import get_platform_info

info = get_platform_info()
print(info.soc_version)   # 例如 "DAV_3510"
print(info.core_num)         # Cube或混合Kernel的上限
print(info.vector_core_num)  # 仅Vector Kernel的上限

max_blocks = info.vector_core_num  # 本例假设Kernel仅包含Vector段
kernel[None, max_blocks](q, k, v, o)
```

**(b) 根据任务量限制逻辑核数**，避免逻辑核数超过Tile数量：

```python
max_blocks = info.vector_core_num  # 仅Vector Kernel
block_dim = min(max_blocks, total_tiles)
kernel[None, block_dim](x, y, z)
```

JIT启动不会自动截断超过平台上限的`block_dim`。仅Vector Kernel使用
`vector_core_num`作为上限，Cube或混合Kernel使用`core_num`。Host侧必须根据Kernel类型
和任务Tile数计算合法的`block_dim`后再启动。

---

## 传递运行时Tiling数据

Kernel可以通过以下三种方式获取shape、Tile数量、循环次数和算子类型等运行时信息：

### 动态shape（`pl.DYNAMIC`）—— 从Tensor获取

在签名中使用`pl.DYNAMIC`声明动态维度，Kernel中通过`tensor.shape[i]`读取运行时值。

```python
@pl.jit(auto_mutex=True)
def add_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    with pl.section_vector():
        num_cores = pl.get_block_num()
        core_id   = pl.get_block_idx()
        m_tile_num = x.shape[0] // TILE_M
        n_tile_num = x.shape[1] // TILE_N
        for i in pl.range(core_id, m_tile_num, num_cores):
            for j in pl.range(0, n_tile_num, 1):
                ...

# Host侧：shape从传入的Tensor获取
add_kernel[None, num_cores](x, y, z)
```

### 标量参数 —— 传一两个运行时值

需要传单个运行时标量（缩放系数、循环边界、flag）时，直接在签名里加一个`pl.DT_*`参数，
在启动时直接传入标量值：

```python
@pl.jit()
def scaled_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_INT8],
    scale_bits: pl.DT_INT32,          # <--运行时标量（预编码的float32位模式）
):
    ...
    pl.store(out, acc, [0, 0], scale=scale_bits)

# Host侧启动
scaled_kernel(a, out, scale_bits)      # 直接传 int
```

### TilingData结构体 —— 传递一组Tiling参数

需要传递多个参数时，可以将其封装为`@dataclass`形式的TilingData类。字段支持
`int`、`float`、`bool`和定长数组`T[N]`。框架会根据字段定义序列化TilingData，
并在Kernel启动时将其传递到Device侧。

```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class AddTiling:
    shape: int[4]       # 每维大小（运行时shape）
    opkind: int[8]      # 算子选择器，值在 opkind[4]

@pl.jit(auto_mutex=True)
def add_dynrank_kernel(
    x: pl.Ptr[pl.DT_FP16],
    y: pl.Ptr[pl.DT_FP16],
    z: pl.Ptr[pl.DT_FP16],
    tiling: AddTiling,             # TilingData参数
):
    N = tiling.shape[3]                          # 字段当普通运行时值用
    M = tiling.shape[0] * tiling.shape[1] * tiling.shape[2]
    tensor_x = pl.make_tensor(x, [M, N])
    ...
    if tiling.opkind[4] == 0:                    # 数组元素驱动运行时分支
        pl.add(tile_c, tile_a, tile_b)

# Host侧构造TilingData，并作为最后一个参数传入
tiling = AddTiling(shape=[1, 1, 513, 513], opkind=[0,0,0,0,1,0,0,0])
add_dynrank_kernel[None, num_cores](x, y, z, tiling)
```

TilingData的字段类型、数组和布局规则参见[TilingData](TilingData.md)。三种运行时参数
传递方式可以组合使用，例如使用`pl.Ptr`作为输入，并通过TilingData传递shape、循环边界
和算子选择器。

<a id="tiling-key--编译期特化而非运行时传值"></a>

### TilingKey —— 编译期特化

动态shape、标量参数和TilingData均用于传递运行时值。若某个模式开关会改变主要代码路径，
并且每种模式需要生成独立的专用Kernel，可以使用TilingKey。声明字段类后，在启动参数
的第三个位置传入具体Key，编译阶段会将字段折叠为常量。

```python
from pypto_pro.runtime.tilingkey import TilingKeyField

class FaTilingKey:
    # 每个字段：占几 bit + 允许的取值
    NeedAttnMask = TilingKeyField(bits=1, values=[0, 1])

    # 可选：拒绝非法组合（key 是按定义顺序的取值元组）
    def is_valid(self, key):
        (need_attn_mask,) = key
        return True

@pl.jit(auto_mutex=True, tiling_key=FaTilingKey)
def fa_kernel(q: pl.Ptr[pl.DT_FP16], k: pl.Ptr[pl.DT_FP16],
              v: pl.Ptr[pl.DT_FP16], o: pl.Ptr[pl.DT_FP16], tiling: OpTiling):
    ...
    if NeedAttnMask == 1:            # 被折叠成常量：另一分支在该 key 下是死代码，直接消除
        ...                          # 载入并应用 mask
    ...

# 启动参数的第三项为Key字典，每个Key对应一个专用Kernel
fa_kernel[None, num_cores, {"NeedAttnMask": 1}](q, k, v, o, tiling)   # causal + mask
fa_kernel[None, num_cores, {"NeedAttnMask": 0}](q, k, v, o, tiling)   # full，同一份源码
```

**运行时参数与TilingKey的选择**

| 需求                                        | 用哪个           |
|---------------------------------------------|------------------|
| 取值频繁变化，需要避免重复编译 | 运行时标量或TilingData |
| 取值有限，并且分支会显著改变生成代码 | TilingKey |
| 需要在启动时拒绝非法组合 | TilingKey与`is_valid()` |

TilingKey可以与运行时参数同时使用，例如使用TilingKey选择代码路径，并通过TilingData
传递运行时shape。完整用法参见[TilingKey](tiling_key.md)。

---

## Tiling设计原则

设计Tiling时，需要综合考虑多核负载、片上存储容量、数据对齐和尾块处理。

### 多核切分：负载均衡

多核切分应使各核任务量尽量均衡，并兼顾数据连续性和Tile粒度。Tile过小会增加调度开销，
Tile过大则会降低可用并行度。

以逐元素算子为例，Host侧可以根据每核最小处理量（如4KB）估算实际使用的核数，避免
小数据量启用过多核：

```python
# Host侧：根据任务量计算实际使用的核数
MIN_ELEMS_PER_CORE = 4 * 1024 // elem_bytes         # 每核至少 4KB
core_by_load = (numel + MIN_ELEMS_PER_CORE - 1) // MIN_ELEMS_PER_CORE
num_cores = min(core_by_load, info.core_num)
```

跨步循环可使各核处理的Tile数量最多相差1。实现步骤如下：

1. Host侧计算总Tile数`total_tiles`和使用的逻辑核数`num_cores`；
2. Kernel中使用`for idx in pl.range(core_id, total_tiles, num_cores)`分配任务。

跨步分配使各核处理的Tile数量最多相差1，从而实现负载均衡。

### UB（片上缓冲区）切分：确定单次处理量

UB切分的要点是：**受UB容量限制（Ascend 950PR/Ascend 950DT为248KB），
单次处理量要对齐到Vector指令友好的粒度（如256B）。**

在PyPTO Pro中，UB切分通过选择Tile shape实现。一个`[TILE_M, TILE_N]`的Tile表示
单次处理的数据块。选择Tile尺寸时应满足以下要求：

- **不超过UB容量**：所有并存Tile的总字节数（包括双缓冲副本）不能超过UB容量。
  每份Tile占用`prod(tile.shape) * dtype_bytes`字节，N缓冲时需乘以N。
- **对齐友好**：`TILE_N`（内层连续维）选成Vector对齐粒度的整数倍（如FP16取128、
  FP32取64等），提升指令效率。
- **处理非整除场景**：Tile尺寸无需整除Tensor shape，可通过向上取整计算Tile数量，
  并使用`set_validshape`处理尾块（参见[尾块处理](tail_block_handling.md)）。

```python
TILE_M, TILE_N = 128, 128                      # 一次处理 128x128 = 16384 元素
tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16,
                        target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
m_tiles = (M + TILE_M - 1) // TILE_M           # 向上取整覆盖尾块
n_tiles = (N + TILE_N - 1) // TILE_N
```

**UB预算的一个具体算例。**逐元素加法`c = a + b`，FP16，三个操作数各双缓冲：

```text
单份 tile 字节 = TILE_M * TILE_N * sizeof(FP16) = 128 * 128 * 2 = 32 KB
buffer 份数    = 3 个操作数(a,b,c) x 2(双缓冲) = 6 份
总占用         = 6 * 32 KB = 192 KB
```

Ascend 950PR/Ascend 950DT的UB容量为248KB，因此192KB满足容量限制。若改用FP32，
单份Tile占用64KB，6份缓冲区共占用384KB，超过UB容量。此时应缩小Tile，例如使用
`128 × 64`，或减少缓冲区份数。所有并存缓冲区的总字节数必须小于或等于UB容量。

### 缓冲区规划：双缓冲重叠

应统一规划输入、输出和中间缓冲区，并使用双缓冲使数据搬运与计算重叠。

在PyPTO Pro里用`pl.make_tile_group` + `auto_mutex=True`实现双 / N缓冲，框架自动插入
同步（细节见[Tensor、Tile与TileGroup](Python_programming_overview.md)）：

```python
a_db = pl.make_tile_group(type=tile_type, addrs=0x0000,  mutex_ids=[0, 1])   # 双缓冲
b_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[2, 3])
c_db = pl.make_tile_group(type=tile_type, addrs=0x20000, mutex_ids=[30, 31])
```

### 分支场景覆盖

实现需要覆盖不同数据类型、shape大小、对齐方式和边界场景：

- **数据类型和模式分支**：使用TilingData字段在Kernel中选择运行时分支，或使用
  TilingKey进行编译期特化。
- **对齐和边界**：使用尾块机制（`valid_shape=[-1,-1]`和`set_validshape`）覆盖非对齐
  shape，无需分别实现对齐和非对齐Kernel。

---

## 完整可执行示例

以下示例演示动态shape、多核和双缓冲的组合使用。示例将`[M, N]`按`128 × 128`
切分为Tile，并沿输出行方向跨步分配给不同逻辑核。

```python
import os
import torch
import pypto_pro.language as pl

TILE_M = 128
TILE_N = 128


@pl.jit(auto_mutex=True)
def add_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16,
                            target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000,  mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x20000, mutex_ids=[30, 31])

    with pl.section_vector():
        num_cores = pl.get_block_num()
        core_id   = pl.get_block_idx()
        m_tile_num = x.shape[0] // TILE_M
        n_tile_num = x.shape[1] // TILE_N

        # 每个核跨步遍历 M 方向的 Tile；内层遍历 N 方向。
        for i in pl.range(core_id, m_tile_num, num_cores):
            for j in pl.range(0, n_tile_num, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])


def test_add():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    torch.npu.set_device(device)
    M_SIZE, N_SIZE = 8192, 4096
    num_cores = M_SIZE // TILE_M           # 每个核一行 Tile；此处能整除
    torch.manual_seed(0)
    x = torch.rand([M_SIZE, N_SIZE], device=device, dtype=torch.float16)
    y = torch.rand([M_SIZE, N_SIZE], device=device, dtype=torch.float16)
    z = torch.empty([M_SIZE, N_SIZE], device=device, dtype=torch.float16)

    add_kernel[None, num_cores](x, y, z)   # 启动语法：[stream, block_dim]
    torch.npu.synchronize()
    torch.testing.assert_close(z, x + y)


if __name__ == "__main__":
    test_add()
```

示例说明：

- `add_kernel[None, num_cores](...)`：`None`表示默认Stream，`num_cores`表示逻辑Block数。
- 对这个仅包含Vector段的Kernel，`pl.get_block_num()` / `pl.get_block_idx()`给出工作单元总数 / 当前工作单元索引。
- `for i in pl.range(core_id, m_tile_num, num_cores)` —— 标准跨步多核切分。
- 这里`M_SIZE`恰好被`TILE_M`整除；不整除时把`num_cores`改成
  `min(info.vector_core_num, ceildiv(M, TILE_M) * ceildiv(N, TILE_N))`并按[尾块文档](tail_block_handling.md)
  处理边界。

### Matmul的多核切分

Matmul的切分思路相同：把输出`[M, N]`按`128 × 128`分块，外层按输出行块跨步分配给Cube逻辑核，内层遍历输出列块：

```python
with pl.section_cube():
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx()
    for i in pl.range(core_id, m // 128, num_cores):
        for j in pl.range(0, n // 128, 1):
            # load_tile -> move -> matmul -> store_tile
            ...
```

完整、可执行的`K=128`示例参见[Matmul算子快速入门](../../quick_start/Matmul_operator_quick_start.md)。`K`大于单个K Tile时，需要沿K轴循环并使用累加语义。

多核、TilingData与尾块处理的组合用法参见[TilingData](TilingData.md)和
[尾块处理](tail_block_handling.md)。

---

<a id="常见坑"></a>

## 使用限制与建议

- **`block_dim`超过对应执行域上限。** Host侧根据Kernel类型选择`vector_core_num`或`core_num`，再与任务Tile数取最小值。
- **混合Kernel的Vector段工作单元数。** 1:2模式下使用`get_block_num() * get_subblock_num()`；`get_block_idx()`直接返回全局AIV逻辑索引。
- **逻辑核数多于Tile数。** 多出的逻辑核不会进入
  `pl.range(core_id, total_tiles, num_cores)`循环，造成资源空转。可使用
  `num_cores = min(core_num, total_tiles)`限制逻辑核数。
- **连续分段导致不均衡。** 手写`[k*chunk:(k+1)*chunk]`且不处理余数时，最后一个核可能显
  著多做或少做。优先用跨步`pl.range(core_id, total, num_cores)`。
- **把TilingData字段当编译期常量用。** TilingData字段是运行时值，不能用在需要编译期
  `int`的地方（例如`TileType`的静态`shape`）；用在`pl.range`、`make_tensor`的
  shape/stride、算术、`if`条件里。
- **TilingData必须是最后一个参数。** TilingData实例必须位于Kernel形参和启动实参的末尾。

---

## 使用摘要

```python
# 仅Cube或仅Vector的Kernel：分配当前核的任务
num_cores = pl.get_block_num()      # 等于启动时传入的block_dim
core_id = pl.get_block_idx()        # 0 .. num_cores-1

total_tiles = m_tiles * n_tiles
for idx in pl.range(core_id, total_tiles, num_cores):
    ...

# 1:2混合Kernel的Vector段：
num_cores = pl.get_block_num() * pl.get_subblock_num()
core_id = pl.get_block_idx()        # 已按subblock展平

# 仅Vector Kernel：设置逻辑Block数
from pypto_pro.runtime.platform import get_platform_info
block_dim = min(get_platform_info().vector_core_num, total_tiles)
kernel[None, block_dim](x, y, z)                  # [stream, block_dim]

# 传递Tiling数据
# (a) 动态shape：签名使用pl.DYNAMIC，Kernel内通过x.shape[0]读取运行时维度
# pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16]
# (b) 标量：在签名中增加pl.DT_*参数，并在启动时传值
def k(a, out, scale: pl.DT_INT32): ...
# (c) TilingData：@dataclass，作为最后一个参数
@dataclass
class T:
    shape: int[4]
    flags: int[8]
tiling = T(shape=[1,1,513,513], flags=[0]*8)
kernel[None, block_dim](x, y, z, tiling)
```
