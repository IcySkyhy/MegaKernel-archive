# FC3XXX-FC5XXX

## FC3000 ERR_PARAM_INVALID

**错误描述**

Matmul内部入参非法：框架内部Shape、Format等参数取值不满足约束。

**可能原因**

NA

**处理方式**

1. 查阅[pypto.matmul](../../../api/operation/pypto-matmul.md)、[pypto.scaled_mm](../../../api/operation/pypto-scaled_mm.md)文档确认输入输出满足要求。
2. 若问题仍未解决，请访问社区提交[Issue](https://gitcode.com/cann/pypto/issues)。

## FC3001 ERR_PARAM_MISMATCH

**错误描述**

Matmul内部入参不匹配：框架内部参数之间维度等不一致。

**可能原因**

NA

**处理方式**

1. 查阅[pypto.matmul](../../../api/operation/pypto-matmul.md)、[pypto.scaled_mm](../../../api/operation/pypto-scaled_mm.md)文档确认输入输出满足要求。
2. 若问题仍未解决，请访问社区提交[Issue](https://gitcode.com/cann/pypto/issues)。

## FC3002 ERR_PARAM_UNSUPPORTED

**错误描述**

Matmul内部入参不支持：框架内部使用了不支持的参数组合。

**可能原因**

- 不满足scale_tensor数据类型约束：scale_tensor非DT_UINT64/DT_INT64；或量化/反量化场景输入输出数据类型不满足（DT_INT8输入输出DT_FP16，或任意输入输出DT_INT8）。

**处理方式**

1. 查阅[pypto.matmul](../../../api/operation/pypto-matmul.md)、[pypto.scaled_mm](../../../api/operation/pypto-scaled_mm.md)文档确认输入输出满足要求。
2. 若问题仍未解决，请访问社区提交[Issue](https://gitcode.com/cann/pypto/issues)。

## FC5000 ERR_RUNTIME_NULLPTR

**错误描述**

Matmul运行时错误：Matmul在运行时出现了空Tensor。

**可能原因**

NA

**处理方式**

1. 确认传入matmul接口的输入输出Tensor均非空且已完成地址分配，确认是否存在nullptr。

   ```python
   # 正确示例-输入Tensor均已分配数据
   a = pypto.tensor([16, 32], pypto.DT_FP16, "a")
   b = pypto.tensor([32, 64], pypto.DT_FP16, "b")
   out = pypto.matmul(a, b, pypto.DT_FP16)
   ```

2. 若问题仍未解决，请访问社区提交[Issue](https://gitcode.com/cann/pypto/issues)。

## FC5002 ERR_RUNTIME_LOGIC

**错误描述**

Matmul运行时逻辑异常：前置校验未通过或计算流程进入异常分支。

**可能原因**

- 用户输入未通过合法校验。
- Matmul内部运行出现了空Tensor。

**处理方式**

1. 查阅[pypto.matmul](../../../api/operation/pypto-matmul.md)、[pypto.scaled_mm](../../../api/operation/pypto-scaled_mm.md)文档确认输入输出满足要求。
2. 若问题仍未解决，请访问社区提交[Issue](https://gitcode.com/cann/pypto/issues)。
