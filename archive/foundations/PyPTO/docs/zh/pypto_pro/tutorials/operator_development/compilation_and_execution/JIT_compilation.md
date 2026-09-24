# AI Core算子JIT编译基本用法

当开发者完成Kernel函数的编写后，通过`@pl.jit()`装饰器即可实现即时编译（JIT），无需手动执行编译命令。

## JIT编译流程

JIT编译的流程如下：

1. 首次调用被`@pl.jit()`装饰的函数时，PyPTO Pro解析函数体中的Tile定义、数据搬运和计算操作，构建计算图
2. 编译器对计算图进行多轮Pass优化，包括算子融合、调度优化、内存重用等
3. 编译器通过CodeGen生成针对NPU的优化代码，并缓存为二进制文件
4. 后续调用时直接加载缓存的二进制文件在NPU上执行，无需重复编译

## 基本用法

以下示例使用[`pl.TileType`](../../../api/SIMD-API/basic_data_structures/TileType.md)定义Tile，通过[`pl.make_tile_group`](../../../api/SIMD-API/operation/resource_management/make_tile_group.md)分配片上缓冲区，并依次调用[`pl.load`](../../../api/SIMD-API/operation/memory_data_movement/load.md)、[`pl.add`](../../../api/SIMD-API/operation/memory_vector_computation/elementwise/add.md)和[`pl.store`](../../../api/SIMD-API/operation/memory_data_movement/store.md)完成数据搬入、计算和搬出。

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.jit(auto_mutex=True)
def add_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16],
    b: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[2])

    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.add(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])

# 首次调用触发JIT编译
device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
device = f"npu:{device_id}"
torch.npu.set_device(device)
a = torch.rand(64, 64, device=device, dtype=torch.float16)
b = torch.rand(64, 64, device=device, dtype=torch.float16)
out = torch.empty(64, 64, device=device, dtype=torch.float16)

# None表示使用PyTorch NPU当前Stream，1表示启动1个AI Core
add_kernel[None, 1](a, b, out)        # 首次启动：编译 + 执行
torch.npu.synchronize()

add_kernel[None, 1](a, b, out)        # 相同编译签名：直接执行
```

## Kernel下发、Stream与同步

JIT编译完成后，Kernel会被提交到NPU执行。启动语法为`kernel[stream, block_dim](...)`：

- `stream`指定任务下发使用的PyTorch NPU Stream，传入`None`表示使用当前Stream。
- `block_dim`指定启动时使用的逻辑Block数量。各执行域中的逻辑AI Core执行同一份Kernel代码，并通过`pl.get_block_idx()`的全局逻辑索引区分数据分片；混合Kernel中AIC/AIV的实际逻辑核数还取决于两者比例。

Kernel下发相对于Host异步执行。Host调用Kernel后会继续执行后续代码，不会自动等待NPU计算完成。因此，在读取输出、进行精度比较或统计Kernel耗时之前，需要同步对应的Stream：

```python
# 使用当前Stream启动Kernel。
add_kernel[None, num_cores](a, b, out)

# 等待当前设备上已下发的任务完成。
torch.npu.synchronize()
result = out.cpu()
```

也可以显式创建并传入Stream，只等待该Stream上的任务：

```python
stream = torch.npu.Stream()
add_kernel[stream, num_cores](a, b, out)
stream.synchronize()
```

同一Stream内的任务按照下发顺序执行。使用不同Stream时，如果任务之间存在数据依赖，需要通过PyTorch NPU的Stream同步机制显式建立依赖，避免后一个任务在前一个任务完成前访问数据。

## 架构指定

通过`arch`参数可以指定目标NPU架构。当前可显式指定`"a5"`，也可以省略该参数以自动检测架构：

```python
# 指定A5架构
@pl.jit(arch="a5")
def kernel_a5(x, out):
    ...

# 自动检测（默认）
@pl.jit()
def kernel_auto(x, out):
    ...
```

| arch值 | 对应产品 |
|:---|:---|
| “a5” | Ascend 950PR/Ascend 950DT |
| None | 自动检测当前受支持设备的架构 |

## 全局配置

除了通过`@pl.jit()`参数为单个Kernel配置编译选项外，还可以在`tile_fwk_config.json`中设置全局默认值。该文件位于`framework/src/interface/configs/tile_fwk_config.json`，修改后需重新编译安装生效。pypto_pro相关的配置项如下：

| 配置项 | 所在段 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `compile_timeout` | `host` | 600 | 编译超时时间（秒），与`@pl.jit(compile_timeout=...)`等价 |

`@pl.jit()`参数优先级高于全局配置：若同时在`tile_fwk_config.json`和`@pl.jit()`中设置了同一参数，以`@pl.jit()`中的值为准。

## 编译产物

JIT编译完成后，编译产物默认缓存在`./build/<kernel_name>__<arch>/`目录下（`<arch>`为目标架构，如`a5`）。每个编译实例均使用独立的TilingKey子目录：使用TilingKey时为`tk_<packed>/`，其中`<packed>`为Key的十六进制打包值；未使用TilingKey时为`tk_none/`。使用datatype特化时，TilingKey子目录位于`dt_<hash>/`下；使用静态签名特化时，基础目录名称还会包含对应的签名后缀。主要产物位于当前编译实例的`tk_<packed>/`或`tk_none/`目录下，包括：

- **kernel.cpp**：CodeGen生成的Device侧C++源码，包含Kernel的计算逻辑实现。
- **call_kernel.cpp / call_kernel.so**：Host侧Launcher源码及其编译后的共享库，负责参数打包和Kernel下发。
- **tiling头文件**（`*_tiling.h`）：当Kernel包含TilingData参数时生成，描述tiling结构体的C布局。
- **kernel.cce.i**：在编译目录下手动执行`bisheng -xcce -DREGISTER_BASE -E -I$ASCEND_TOOLKIT_HOME/include -I$ASCEND_HOME_PATH --cce-aicore-arch=dav-c310 kernel.cpp > kernel.cce.i`生成的宏展开CCE源码，便于调试宏展开问题。

> [!NOTE]说明
> 默认情况下，编译成功后`kernel.cpp`和`call_kernel.cpp`等中间源文件会保留在产物目录中，便于调试。若需查看Device侧生成的代码，可直接阅读`kernel.cpp`。
