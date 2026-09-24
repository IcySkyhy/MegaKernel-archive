# pypto.scatter

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持
<!-- end id3 -->

## 函数原型

```python
scatter(input: Tensor, dim: int, index: Tensor, src: Union[float, Element, Tensor], *, reduce: str = None) -> Tensor
```

scatter\_的non-inplace版本，可参考[pypto.scatter\_](pypto-scatter_.md)

## 参数说明

请参考[pypto.scatter_](pypto-scatter_.md)的参数说明。

## 返回值说明

请参考[pypto.scatter_](pypto-scatter_.md)的返回值说明。

## 约束说明

1. 请参考[pypto.scatter_](pypto-scatter_.md)的约束说明。
2. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

请参考[pypto.scatter_](pypto-scatter_.md)的调用示例。
