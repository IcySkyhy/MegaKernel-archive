# Tensor的创建

Tensor是PyPTO中的基本数据结构，用于表示将在计算图中使用并在NPU上执行的多维数组。

在PyPTO中，Tensor表示其数据的结构和属性，这使得PyPTO能够构建计算图，并在执行前对其进行优化。Tensor在执行时才包含实际值，未初始化的Tensor中的值都是随机的，在执行时需要按需初始化。

## 创建Tensor

- 创建基础Tensor

    ```python
    #创建形状为[2, 3]、数据类型为FP16的Tensor
    tensor = pypto.tensor([2, 3], pypto.DT_FP16, "my_tensor")
    ```

    参数说明：

    - shape：维度，支持整数列表。
    - dtype：表示Tensor中存储的数据类型，支持DataType类型，例如，DT\_FP16表示16位半精度浮点数。
    - name：名称，支持字符串类型，可选。但建议为Tensor提供有意义的名称，以便于调试和理解计算图结构。
    - format：数据排布格式，支持TileOpFormat类型，可选，默认为：TILEOP\_ND。
                format显式标记时,性能更优,要求传入的torch tensor与pypto.Tensor声明的format一致;

- 创建带格式的Tensor

    ```python
    #使用NZ格式创建一个Tensor
    tensor = pypto.tensor([-1, 32], pypto.DT_FP16, "nz_tensor", pypto.TileOpFormat.TILEOP_NZ)
    ```

    支持的格式：

    - TILEOP\_ND：ND格式，N维数组，在PyPTO中采用行优先模式。
    - TILEOP\_NZ：NZ格式，矩阵乘相关的特殊格式。二维矩阵被分为若干个分形（分形大小更能适配一次Cube计算），分形按照列优先即N字形排布；每个分形按照行优先即Z字形排布。详细介绍请参见[数据排布格式](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0099.html)。

- 在子函数内创建Tensor并返回至主函数

    ```python
    def sub_function():
        #创建形状为[2, 3]、数据类型为FP16的Tensor
        tensor = pypto.tensor([2, 3], pypto.DT_FP16, "my_tensor")
        return tensor

    def main_function():
        sub_tensor = sub_function()
    ```

- PyTorch的Tensor转换为PyPTO的Tensor

    ```python
    # prepare data
    input_data = torch.rand(shape, dtype=torch.float, device='npu')
    output_data = torch.zeros(shape, dtype=torch.float, device='npu')

    #convert from torch tensor to pypto tensor
    pto_input = pypto.from_torch(input_data, "in_0")
    pto_output = pypto.from_torch(output_data, "out_0")
    ```

## 查看Tensor属性

Tensor包括形状（shape）、数据类型（dtype）、数据排布格式（format）、维数（dim）、名称（name）等基本属性，通过pypto.tensor相关操作接口可以查询这些属性信息。

```python
tensor = pypto.tensor([2,3, 4], pypto.DT_FP16, "example")

#形状
print(tensor.shape)  #[2, 3, 4]

#数据类型
print(tensor.dtype)  #数据类型.DT_FP16

#维数
print(tensor.dim)    # 3

#格式
print(tensor.format)  #TILEOP_ND

#名称
print(tensor.name)    # "example"
tensor.name = "new_name"  #可以更改
```

## 动态维度Tensor的处理

在实际应用场景，Tensor通常是个可变长的数据。可通过以下方法定义动态Shape的Tensor，并通过-1标记动态维度：

```python
tensor = pypto.tensor([-1, 32], pypto.DT_FP16, "dynamic")

#打印tensor的维度，SymbolicScalar表示当前Shape为符号化标量
print(tensor.shape)
>>> [SymbolicScalar(RUNTIME_GetInputShapeDim(ARG_dynamic,0)), 32]
```

通过如下方法可以获取动态维度的符号化标量，在运行时获取具体数值：

```python
b = pypto.symbolic_scalar(tensor_shape[0])
```

如果Tensor继承自PyTorch Tensor，可以通过pypto.from\_torch接口的参数dynamic\_axis = \[int\]来定义动态维度的Tensor。

```python
# prepare data
input_data = torch.rand(shape, dtype=torch.float, device='npu')
output_data = torch.zeros(shape, dtype=torch.float, device='npu')

#convert from torch tensor to pypto tensor with dynamic axis
pto_input = pypto.from_torch(input_data, "in_0", dynamic_axis=[0])
pto_output = pypto.from_torch(output_data, "out_0", dynamic_axis=[0])
```
