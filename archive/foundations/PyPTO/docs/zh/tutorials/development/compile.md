# 编译与执行

PyPTO通过函数定义在NPU硬件上构建可编译的计算图结构，并利用@pypto.frontend.jit装饰器实现即时编译（JIT），从而充分发挥NPU的并行计算能力，从而提升算子执行效率。

## Kernel函数定义

在执行JIT编译前，需要定义Kernel函数，获取输入输出Tensor、配置Tiling信息并实现计算逻辑。

- 基础函数定义：

    ```python
    def add_kernel(input: pypto.Tensor, out: pypto.Tensor):
        # Tiling setting
        pypto.set_vec_tile_shapes(1, 4, 1, 64)
        out[:] = input + 1
    ```

- 多输入输出函数定义：

    ```python
    def add_kernel(input0: pypto.Tensor, input1: pypto.Tensor, out: pypto.Tensor):
         # Tiling setting
         pypto.set_vec_tile_shapes(1, 4, 1, 64)
         out[:] = input0 + input1
    ```

## JIT编译

当通过PyPTO函数完成kernel的计算流及数据流的编写，可以加上pypto.frontend.jit的装饰器，标记该函数为JIT编译目标，触发PyPTO的编译流程。

```python
@pypto.frontend.jit
def add_kernel(
    input0: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
    input1: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
    out: pypto.Tensor((1, 4, 1, 64), pypto.DT_FP32),
):
     # Tiling setting
     pypto.set_vec_tile_shapes(1, 4, 1, 64)
     out[:] = input0 + input1
```

JIT编译流程为：

- 首次调用时，函数以“记录模式”执行，操作被记录并优化为计算图，经编译器生成针对NPU的优化代码后缓存二进制文件；
- 后续调用时，函数直接调用缓存的二进制文件在NPU上执行，无需重复编译。

完整样例请参考：[hello_world](../../../../examples/00_hello_world/hello_world.py)。

## 条件编译

JIT装饰器支持参数配置，可根据配置支持不同的条件编译：

```python
@pypto.frontend.jit(
    host_options={},
    pass_options={},
    runtime_options={},
    verify_options={},
    debug_options={}
)
def advanced_function(input0, input1):
    # 实现自定义计算逻辑
    pass
```

JIT配置选项说明如下：

- codegen\_options：代码生成设置。
- host\_options：主机端选项。
- pass\_options：编译器PASS传递选项。
- runtime\_options：运行时执行选项。
- verify\_options:精度检验工具选项。
- debug\_options:性能数据采集功能配置选项。

除了使用JIT装饰器启用不同配置外，还可以直接在代码中调用pypto.set\_codegen\_options、pypto.set\_host\_options、pypto.set\_pass\_options、pypto.set\_runtime\_options，pypto.set\_verify\_options, pypto.set\_debug\_options接口进行配置，例如：

```python
pypto.set_codegen_options(support_dynamic_aligned=True)
```

建议优先使用JIT入参配置各类选项，因为JIT配置选项提供了配置的便利性，同时避免在计算函数内部出现与数据流和计算不相关的代码。

## 定义多个JIT函数

您可以定义多个JIT函数并将它们一起使用：

```python
def add_core(input0: pypto.Tensor, input1: pypto.Tensor, output: pypto.Tensor, val: int, add1_flag: bool = False):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    if add1_flag:
        t3 = input0 + input1
        output[:] = t3 + val
    else:
        output[:] = input0 + input1

@pypto.frontend.jit
def add_kernel_true(
    input0: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    input1: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    val: int
):
    add_core(input0, input1, output, val, True)


@pypto.frontend.jit
def add_kernel_false(
    input0: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    input1: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    val: int
):
    add_core(input0, input1, output, val, False)


#使用这两个函数
def add_add1flag_false(input_data0, input_data1, val=0):
    output_data = torch.empty_like(input_data0)
    add_kernel_false(input_data0, input_data1, output_data, val)
    return output_data

def add_add1flag_true(input_data0, input_data1, val=0):
    output_data = torch.empty_like(input_data0)
    add_kernel_true(input_data0, input_data1, output_data, val)
    return output_data

add_add1flag_false(input_data0, input_data1, val)
add_add1flag_true(input_data0, input_data1, val)
```

完整样例请参考：[multi_jit.py](../../../../examples/03_advanced/patterns/function/multi_jit.py)
