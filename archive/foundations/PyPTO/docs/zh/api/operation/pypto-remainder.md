# pypto.remainder

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

## 功能说明

将input的每个元素和other中对应位置的元素进行取余运算，计算公式如下：

$$
res_i = input_i - other_i * floor(input_i / other_i)
$$

## 函数原型

```python
remainder(
    input: Union[Tensor, int, float],
    other: Union[Tensor, int, float],
    precision_type: PrecisionType = PrecisionType.HIGH_PRECISION
) -> Tensor:
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| input  | 输入      | 源操作数。<br>支持的类型为：Tensor、int、float。<br>Tensor支持的数据类型为：DT_FP32，DT_FP16，DT_BF16，DT_INT32，DT_INT16。<br>不支持空Tensor；Shape仅支持1-4维，支持多维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |
| other  | 输入      | 源操作数。<br>支持的类型为：Tensor、int、float。<br>Tensor支持的数据类型为：DT_FP32，DT_FP16，DT_BF16，DT_INT32，DT_INT16。<br>不支持空Tensor；Shape仅支持1-4维，支持多维度广播到相同形状；Shape Size不大于2147483647（即INT32_MAX）。 |
| precision_type | 输入      | 精度模式枚举类型，用以控制取余计算的精度模式，具体定义为：[PrecisionType](../datatype/PrecisionType.md)。<br>默认为HIGH_PRECISION（高精度模式）。 |

## 返回值说明

返回输出Tensor，Shape为input和other广播后大小，数据类型和输入Tensor的数据类型相同。

## 约束说明

1. 当前不支持混合精度类型输入，即输入都是Tensor时数据类型都相同，输入有一个是标量时，Tensor的数据类型必须是对应的整数类型（DT_INT32或DT_INT16）或浮点数类型（DT_FP32、DT_FP16、DT_BF16）；
2. 当input为整型数据类型时，**other不能含0**，整数取余的结果由芯片类型决定，可能为0或-1。
3. 若输入Tensor的数据类型为DT_INT32，数据范围超过\[-2^24, 2^24\]范围时不保证精度；
4. precision_type使用说明：
   <!-- npu="950" id4 -->
   - Ascend 950PR/Ascend 950DT：支持
   <!-- end id4 -->
   <!-- npu="A3" id5 -->
   - Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持，默认使用指令模式 `INTRINSIC`
   <!-- end id5 -->
   <!-- npu="910b" id6 -->
   - Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持，默认使用指令模式 `INTRINSIC`
   <!-- end id6 -->
5. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

如非广播场景，输入input shape为[m, n]，other为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

广播场景，输入input shape为[m, n]，other为[m, 1]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
a = pypto.tensor([7.0, 8.0, 9.0], pypto.DT_FP32)
b = pypto.tensor([-3.0, -3.0, -3.0], pypto.DT_FP32)
out = pypto.remainder(a, b)
```

结果示例如下：

```python
输入数据a:    [7.0, 8.0, 9.0]
输入数据b:    [-3.0, -3.0, -3.0]
输出数据out:  [-2.0, -1.0, 0.0]
```

### 高精度模式示例

```python
a = pypto.tensor([7.0, 8.0, 9.0], pypto.DT_FP16)
b = pypto.tensor([-3.0, -3.0, -3.0], pypto.DT_FP16)
out = pypto.remainder(a, b, pypto.PrecisionType.HIGH_PRECISION)
```

### 指令模式示例

```python
a = pypto.tensor([7.0, 8.0, 9.0], pypto.DT_FP32)
b = pypto.tensor([-3.0, -3.0, -3.0], pypto.DT_FP32)
out = pypto.remainder(a, b, pypto.PrecisionType.INTRINSIC)
```
