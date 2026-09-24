# Add算子快速入门

本示例是一个入门实践，基于PyPTO Pro SIMD实现Add算子，帮助您快速上手。它完整呈现了Kernel函数定义、Tile配置、数据搬运、计算及运行的全流程，助您建立整体认知。开始前，请先参考[环境准备](../../../install/prepare_environment.md)完成基础环境搭建。

## Add算子

**功能介绍**：Add算子的数学表达式为$z = x + y$，计算逻辑为逐元素完成两个张量的加法。

## 算子设计

| 模块 | 说明 |
|:---|:---|
| Kernel函数定义 | 通过`@pl.jit(auto_mutex=True)`声明JIT编译目标，开启自动同步 |
| Tile定义 | 使用[`pl.TileType`](../../api/SIMD-API/basic_data_structures/TileType.md)定义片上Tile的形状、数据类型和目标内存空间 |
| Tile分配 | 使用[`pl.make_tile_group`](../../api/SIMD-API/operation/resource_management/make_tile_group.md)分配片上内存，通过`mutex_ids`指定互斥缓冲，框架自动插入同步 |
| 数据搬入 | 通过[`pl.load`](../../api/SIMD-API/operation/memory_data_movement/load.md)将GM数据搬入UB Tile |
| 数据计算 | 通过[`pl.add`](../../api/SIMD-API/operation/memory_vector_computation/elementwise/add.md)完成逐元素加法 |
| 数据搬出 | 通过[`pl.store`](../../api/SIMD-API/operation/memory_data_movement/store.md)将UB Tile结果写回GM |

## 算子代码实现

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.jit(auto_mutex=True)
def add_kernel(a: pl.Tensor[[64, 64], pl.DT_FP16], b: pl.Tensor[[64, 64], pl.DT_FP16],
               out: pl.Tensor[[64, 64], pl.DT_FP16]):
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


# Host端调用
device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
device = f"npu:{device_id}"
torch.npu.set_device(device)
torch.manual_seed(0)
a = torch.rand(64, 64, device=device, dtype=torch.float16)
b = torch.rand(64, 64, device=device, dtype=torch.float16)
out = torch.empty(64, 64, device=device, dtype=torch.float16)

add_kernel(a, b, out)
torch.npu.synchronize()

torch.testing.assert_close(out, a + b, rtol=1e-2, atol=1e-2)
print("Add kernel passed!")
```

> [!NOTE]说明
>
> - `pl.Tensor[[64, 64], pl.DT_FP16]`中的`[64, 64]`为张量形状，`pl.DT_FP16`为数据类型。
> - `pl.TileType`的`target_memory=pl.MemorySpace.Vec`表示Tile分配在Vector核的UB上。
> - `pl.make_tile_group`通过`mutex_ids`分配缓冲，框架在`auto_mutex=True`时自动插入同步，开发者无需手写`sync_src`/`sync_dst`。
> - `tile_group.current()`获取当前可用缓冲。
> - `pl.section_vector()`标记后续代码在Vector流水单元上执行。
> - 昇腾NPU对FP16和BF16有原生硬件加速，建议在算子开发中优先考虑这些数据类型。
> - 如需进一步了解PyPTO Pro的SIMD编程模型，请参阅[编程范式概述](../programming_paradigm/programming_paradigm_overview.md)。
