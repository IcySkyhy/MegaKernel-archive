# 算子编译报堆栈溢出错误

## 问题现象描述

算子编译时出现类似如下报错字样：

```text
error: stack frame size (*****) exceeds limit (32768) in function '*****'
```

在打屏日志中呈现样例如下：

```text
error: stack frame size (47928) exceeds limit (32768) in function 'TENSOR_nLoop_Unroll1_PATH0_6_0_4503599627370496'
error: stack frame size (47928) exceeds limit (32768) in function 'TENSOR_nLoop_Unroll1_PATH0_6_0_4503599627370496'
2 errors generated.
terminate called after throwing an instance of 'npu::tile_fwk::Error'
  what():  ASSERTION FAILED: ret == 0
CompileCCE failed. errCode = 256, cce file: output/output_20251111_175724_806073/kernel_aicore/TENSOR_nLoop_Unroll1_PATH0_6_17699850674043372772_0_aic.cpp
```

## 原因分析

由于硬件限制，昇腾AI处理器核内的标量处理单元栈空间最大为32K。因此，在算子编译时，底层的毕昇编译器会对算子核内函数的栈使用情况进行分析和校验。如果函数实现较为复杂，例如变量较多或变量生命周期较长等因素，均可能影响分析结果。如果最终分析结果超出32K限制，毕昇编译器将拦截并报告该错误。

在PTO编程模型下，算子核内函数实现复杂的主要原因主要如下：

- 子图规模较大，导致变量过多。
- 子图计算逻辑复杂，导致变量的生命周期较长。

## 解决措施

针对上述可能的原因，可以采取以下措施进行处理：

- 调整Tensor的Tiling切分策略，采用更大的Tile块进行切分。通过增加单个Tile块的计算数据量，在总数据量保持不变的情况下，可以减少子图所需的计算步骤，从而有效减小子图的规模。相关设置接口为：[pypto.set_vec_tile_shapes](../../../api/config/pypto-set_vec_tile_shapes.md)和 [pypto.set_cube_tile_shapes](../../../api/config/pypto-set_cube_tile_shapes.md)。
- 对于矩阵乘（MATMUL）计算场景，建议用户对K轴进行多核切分。
- 修改控制子图大小的选项`cycle_upper_bound`，将单个子图的最大规模限制在指定范围内。相关设置接口为：[pypto.set_pass_options](../../../api/config/pypto-set_pass_options.md)。
