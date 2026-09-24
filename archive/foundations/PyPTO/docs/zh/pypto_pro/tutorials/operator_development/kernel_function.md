# Tile核函数

Tile核函数（Kernel Function）是在NPU设备侧执行的Python函数。它由Host端代码调用，PyPTO Pro框架自动将其编译为硬件指令，并调度到AI Core上执行。每个Kernel函数通过显式的Tile定义、数据搬运和同步控制，精确管理片上计算流程。

## 核函数的定义

定义Tile核函数时需要遵循以下规则：

### 使用JIT装饰器

必须使用`@pl.jit()`装饰器标识该函数为Kernel函数，PyPTO Pro框架会将其编译为NPU可执行的二进制。

### 参数类型标注

Tensor输入输出通常使用[`pl.Tensor`](../../api/SIMD-API/basic_data_structures/Tensor.md)标注，并指定张量的形状和数据类型：

```python
@pl.jit()
def my_kernel(x: pl.Tensor[[64, 64], pl.DT_FP16],
              y: pl.Tensor[[64, 64], pl.DT_FP16],
              out: pl.Tensor[[64, 64], pl.DT_FP16]):
    ...
```

根据数据和参数的组织方式，Kernel还支持以下参数类型：

| 参数类型 | 典型用途 |
| --- | --- |
| `pl.Tensor[[shape], dtype]` | shape和数据类型可在签名中确定的Tensor |
| [`pl.Ptr[dtype]`](../../api/SIMD-API/basic_data_structures/Ptr.md) | 裸指针输入输出；常与TilingData配合重建动态shape的Tensor视图 |
| `pl.DT_*` | 运行时标量参数，例如`pl.DT_INT64`、`pl.DT_FP32` |
| TilingData类 | 传递shape、循环边界、算子选择器等结构化运行时参数 |

下面的示例通过`pl.Ptr`和TilingData重建动态shape的Tensor视图。当前JIT要求
TilingData位于Kernel形参和启动实参的末尾：

```python
from dataclasses import dataclass

@dataclass
class AddTiling:
    m: int
    n: int

@pl.jit(auto_mutex=True)
def dynamic_kernel(
    x: pl.Ptr[pl.DT_FP16],
    out: pl.Ptr[pl.DT_FP16],
    scale: pl.DT_FP32,
    tiling: AddTiling,
):
    tensor_x = pl.make_tensor(x, [tiling.m, tiling.n])
    tensor_out = pl.make_tensor(out, [tiling.m, tiling.n])
    ...
```

核函数不支持返回值，计算结果通过与`pl.Tensor`或`pl.Ptr`输出参数对应的缓冲区写回。

### Tile定义与分配

核函数内部使用[`pl.TileType`](../../api/SIMD-API/basic_data_structures/TileType.md)定义Tile类型，并通过[`pl.make_tile_group`](../../api/SIMD-API/operation/resource_management/make_tile_group.md)等接口分配片上内存：

```python
tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
tile_x = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
tile_y = pl.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[1])
tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[2])
```

### 流水段与同步

计算逻辑需要放在[`pl.section_vector()`](../../api/SIMD-API/operation/controlflow/section_vector_section_cube.md)上下文中，开启`auto_mutex=True`后，搬运与计算间的流水同步由框架按Tile的mutex自动插入：

```python
with pl.section_vector():
    cur_x = tile_x.current()
    cur_y = tile_y.current()
    cur_out = tile_out.current()
    pl.load(cur_x, x, [0, 0])
    pl.load(cur_y, y, [0, 0])
    pl.add(cur_out, cur_x, cur_y)
    pl.store(out, cur_out, [0, 0])
```

### 其他规则

- 运行时标量形参使用`pl.DT_*`类型标注，Host侧传入对应的Python标量值。
- 使用TilingData时，必须将其放在Kernel形参和启动实参的末尾。
- 可以使用`auto_mutex=True`参数启用自动互斥锁插入。

## 核函数的调用

PyPTO Pro中核函数通过方括号启动语法发起，方括号内指定Stream和`block_dim`（逻辑Block数）：

```python
import os
# 准备输入数据
device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
device = f"npu:{device_id}"
a = torch.rand(64, 64, device=device, dtype=torch.float16)
b = torch.rand(64, 64, device=device, dtype=torch.float16)
out = torch.empty(64, 64, device=device, dtype=torch.float16)

# 方括号启动语法：[stream, block_dim]
# None表示默认Stream，num_cores指定逻辑Block数
add_kernel[None, num_cores](a, b, out)
torch.npu.synchronize()
```

Kernel执行域决定启动类型：

- 仅包含`pl.section_vector()`的Kernel编译并启动为Vector（AIV）Kernel，`block_dim`配置AIV逻辑核数。
- 仅包含`pl.section_cube()`的Kernel编译并启动为Cube（AIC）Kernel。
- 同时包含`pl.section_vector()`和`pl.section_cube()`的Kernel编译并启动为混合Kernel。

PyPTO Pro JIT不会自动截断超过平台核数上限的`block_dim`。Host侧应根据Kernel类型和
[`get_platform_info()`](tile_based_python_programming/multi_core_partitioning_and_Tiling.md#在启动时设置逻辑block数block_dim)
返回的核数上限计算`block_dim`。

也可以省略方括号直接调用，此时使用默认`block_dim=1`：

```python
add_kernel(a, b, out)
torch.npu.synchronize()
```

核函数的调用是异步的。首次调用时触发JIT编译，后续调用直接执行缓存的二进制文件，无需重复编译。

### stream的含义与设置

`stream`指定Kernel下发的NPU执行流。当前仅支持传入`None`，表示使用当前Stream：

```python
kernel[None, num_cores](x, out)
```

显式传入`torch.npu.Stream`暂未验证，不建议使用。

### blockDim的含义与设置

`block_dim`为逻辑核数，JIT在launch前会将其clamp到平台可用核数上限：若`block_dim`超过上限则告警并截断。上限取决于编译产物的核类型，由`is_aiv_only`自动判定：

- **纯Vector算子**（kernel仅含`section_vector`、不含`section_cube`）：编译为aiv-only（arch选`dav-*-vec`）。`block_dim`上限为`vector_core_num`，语义为启动多少个AIV实例。例如某处理器有40个AIV核，建议设为40。
- **融合算子**（kernel同时含`section_cube`与`section_vector`）：编译为mix（arch选`dav-c220`/`dav-c310`）。`block_dim`上限为`core_num`，语义为启动多少个AICore组合。运行时每个AICore按1:2拆分为1个AIC与2个AIV，例如20个AICore的处理器建议设为20，实际占用20个AIC与40个AIV。

## Tiling参数化

`@pl.jit()`支持通过`tiling_key`参数实现Tiling参数化，在启动时通过字典选择不同的Kernel实例化，使每种模式各编一份专用Kernel（消除死分支、拿到最优指令）：

```python
from pypto_pro.runtime.tilingkey import TilingKeyField

class MyTilingKey:
    NeedAttnMask = TilingKeyField(bits=1, values=[0, 1])

@pl.jit(tiling_key=MyTilingKey)
def my_kernel(x: pl.Tensor[[64, 64], pl.DT_FP16], out: pl.Tensor[[64, 64], pl.DT_FP16]):
    ...

# 启动时通过字典选择实例化
my_kernel[None, num_cores, {"NeedAttnMask": 1}](x, out)
my_kernel[None, num_cores, {"NeedAttnMask": 0}](x, out)
```

`tiling_key`的完整说明（字段定义、`is_valid`校验、与TilingData的组合、运行时标志与TilingKey的选型对照表）请参考[多核切分与Tiling](tile_based_python_programming/multi_core_partitioning_and_Tiling.md#tiling-key--编译期特化而非运行时传值)。

## JIT配置选项

`@pl.jit()`装饰器支持以下配置选项：

| 选项 | 说明 | 默认值 |
|:---|:---|:---|
| arch | 目标架构，当前可选“a5”；None为自动检测当前受支持设备的架构 | None |
| auto_mutex | 是否启用自动互斥锁插入 | True |
| compile_timeout | 编译超时时间（秒） | 600 |
| name | 自定义Kernel名称，用于构建产物路径隔离 | None |
| tiling_key | Tiling键类型，用于Tiling参数化 | None |
| pipeline | PipelineConfig，用于自动预取流水变换 | None |
| datatype | 数据类型特化，用于同一Kernel支持多种数据类型 | None |

```python
@pl.jit(arch="a5", auto_mutex=True, compile_timeout=200)
def my_kernel(x: pl.Tensor[[64, 64], pl.DT_FP16],
              out: pl.Tensor[[64, 64], pl.DT_FP16]):
    ...
```

> [!NOTE]说明
> 建议优先使用JIT入参配置各类选项，避免在计算函数内部出现与数据流和计算不相关的代码。
