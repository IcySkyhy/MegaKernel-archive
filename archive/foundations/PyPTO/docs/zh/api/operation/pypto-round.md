# pypto.round

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

将输入Tensor的元素四舍五入到指定的位数，若该值与指定位数上的两个小数距离一样，则取指定位数上为偶数。

## 函数原型

```python
round(input: Tensor, decimals: int) -> Tensor
```

## 参数说明

| 参数名      | 输入/输出 | 说明                                                                                                                                                             |
|----------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| input    | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32，DT_FP16，DT_BF16，DT_INT32，DT_INT16。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |
| decimals | 输入      | 源操作数，四舍五入到的小数位数。<br>int类型。                                                                                                                                  |

## 返回值说明

返回Tensor类型。其Shape、数据类型与输入Tensor一致，其元素为输入Tensor对应元素四舍五入到指定位数的结果。

## 约束说明

1. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

## 调用示例

### TileShape设置示例

说明：调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输出一致。

示例1：输入input shape为[m, n]，输出为[m, n]，TileShape设置为[m1, n1]，则m1，n1分别用于切分m，n轴。

```python
pypto.set_vec_tile_shapes(4, 16)
```

### 接口调用示例

```python
x = pypto.tensor([2, 2], pypto.DT_FP32)
y = pypto.round(x, decimals=1)
```

结果示例如下：

```python
输入数据x: [[1.21, 2.35], [3.65, 4.76]]
输出数据y: [[1.2, 2.4], [3.6, 4.8]]
```
