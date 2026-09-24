# HelloWorld

本入门示例基于PyPTO Pro SIMD实现Hello World算子，带你快速上手实践，涵盖Kernel函数定义、JIT编译以及运行的完整流程，帮助开发者建立整体认知。开始前请参考[环境准备](../../../install/prepare_environment.md)完成基础环境搭建。

## Hello World

**功能介绍**：在NPU上打印`Hello World!!!`，验证PyPTO Pro的基本开发流程。

## Kernel函数实现

通过`@pl.jit()`装饰器定义Kernel函数，使用[`pl.printf`](../../api/Utils-API/debugging/printf.md)在Device端打印字符串。[`pl.section_vector()`](../../api/SIMD-API/operation/controlflow/section_vector_section_cube.md)用于声明该段代码在Vector核上执行，其中`pl.printf`由Scalar流水执行。

```python
import pypto_pro.language as pl

@pl.jit()
def hello_world_kernel(out: pl.Tensor[[1], pl.DT_INT32]):
    with pl.section_vector():
        pl.printf("Hello World!!!\n")
        pl.setval(out, 0, 1)
```

> [!NOTE]说明
>
> - PyPTO Pro的Kernel函数通过`@pl.jit()`装饰器标记为JIT编译目标。首次调用时触发编译，后续调用直接执行缓存的二进制文件，无需重复编译。
> - `pl.printf`通过设备侧打印机制输出，具体查看位置由运行环境的CANN日志配置决定。`printf`仅用于调试，生产环境应移除。

## Host端调用

Host端通过PyTorch张量准备输入输出数据，直接调用Kernel函数完成计算。

```python
import os
import torch
import torch_npu

device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
device = f"npu:{device_id}"
torch.npu.set_device(device)

out = torch.zeros(1, device=device, dtype=torch.int32)

hello_world_kernel(out)
torch.npu.synchronize()

print(f"kernel finished, out[0] = {out[0].item()}")
```

## 编译与运行

将上述代码保存为`hello_world.py`，执行：

```bash
python3 hello_world.py
```

运行后，Host端输出`kernel finished, out[0] = 1`。`Hello World!!!`由设备侧打印机制输出，请按当前部署环境的CANN日志配置查看；输出位置不固定为Host标准输出。

> [!NOTE]说明
>
> 如需进一步了解PyPTO Pro的SIMD编程模型，请参阅[编程范式概述](../programming_paradigm/programming_paradigm_overview.md)。
