# ffn_shared_expert_quant算子NPU上板调试案例

## 任务与目标

ffn\_shared\_expert\_quant算子对应GLM4.5网络中MoE共享专家的计算逻辑，包含symmetric\_quantization\_per\_token、matmul、dequant\_dynamic和swiglu，用于进行单个共享专家的量化前向传播计算，通过在不同任务或数据流之间复用同一组权重参数，以学习通用的特征表示，同时减少模型的参数总量。

下面通过该算子介绍功能调试的大致步骤，完整样例请参考：[glm_ffn_shared_expert_quant](../../../../models/glm_v4_5/glm_ffn_shared_expert_quant.py)。

## 问题定位

假设在NPU上板运行时，ffn\_shared\_expert\_quant算子测试用例运行失败并报错，此时需要进行调试，定位问题。

首先可以通过查看Tensor Graph进行问题的定位，主要步骤如下：

1. 按照[开启调试模式](debug.md)所述步骤开启调试模式，重新运行用例后获取ffn\_shared\_expert\_quant算子计算图。
2. 按照[查看计算图](debug.md)所述，使用PyPTO Toolkit可视化工具打开Tensor Graph阶段的计算图文件，例如：Before\_004\_ExpandFunction\_TENSOR\_share\_loop\_idx\_Unroll1\_PATH0\_4.json：

    ![](../figures/fnn_0.png)

3. 实际的算子代码中调用了Matmul接口进行操作，然而通过上图可以看到，无Matmul操作节点，并且后续所有Tensor以及Operation节点都未能加载出来，有明显异常。

    检查算子代码中各个Matmul操作是否正确使用，发现由于第一个Matmul输入的A/B矩阵位置错误，导致A/B矩阵的reduce轴没有满足相等的约束。错误存在于：

    ```python
    up_proj = pypto.matmul(w13, hidden_states_quant, pypto.DT_INT32)
    ```

除了利用计算图进行调试外，也可以根据内部DFX校验机制来定位问题。

以下是ffn\_shared\_expert\_quant算子测试用例运行失败时展示的ERROR信息：

```text
ERROR:root:Record function share_expert_moe_main failed: ASSERTION FAILED: kSizeA == kSizeB
Matrix K dimemsion mismatch, kSizeA: 384, kSizeB: 8
, func ConstructTensorGraph, file cube_operation_impl.cpp, line 1220
libtile_fwk_interface.so(npu::tile_fwk::Tensor npu::tile_fwk::Matrix::ConstructTensorGraph<false, false, false>(npu::tile_fwk::DataType, npu::tile_fwk::Tensor const&, npu::tile_fwk::Tensor const&, npu::tile_fwk::Tensor const&, npu::tile_fwk::Matrix::MatmulExtendParam const&)+0x25d) [0x7fe34630ad3d]
libtile_fwk_interface.so(npu::tile_fwk::Tensor npu::tile_fwk::Matrix::Matmul<false, false, false>(npu::tile_fwk::DataType, npu::tile_fwk::Tensor const&, npu::tile_fwk::Tensor const&)+0x14e) [0x7fe34630b51e]
```

获取关键信息"Matrix K dimemsion mismatch"，得知错误由某Matmul操作传入的Tensor Shape的K轴不相等引起。

## 解决方案

修改该算子的实现代码：

```python
up_proj = pypto.matmul(w13, hidden_states_quant, pypto.DT_INT32)
```

修改结果如下：

```python
up_proj = pypto.matmul(hidden_states_quant, w13, pypto.DT_INT32)
```

再次运行ffn\_shared\_expert\_quant算子测试用例，能够正常通过，问题解决。
