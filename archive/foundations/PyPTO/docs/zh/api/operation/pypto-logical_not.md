# pypto.logical\_not

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

输入Tensor中的0对应转换为True，非0值转换为False。

## 函数原型

```python
logical_not(input: Tensor) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 源操作数。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32，DT_FP16，DT_BF16，DT_BOOL，DT_INT8，DT_UINT8，DT_INT16，DT_UINT16，DT_INT32，DT_UINT32。<br>不支持空Tensor；Shape仅支持1-4维；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回输出Tensor，Tensor的数据类型为DT\_BOOL，Shape与源操作数input Shape相同。

## 约束说明

1. TileShape与input维度保持一致；
2. 由于存在临时内存使用，当输入数据类型为DT\_FP32，TileShape大小有额外约束，假设TileShape为\[a,b,c,d\]，那么a\*b\*c\*d\*sizeof\(input\) + a\*b\*c\*d\*sizeof\(BOOL\) + 20.25KB<UB。其他输入数据类型应该满足，a\*b\*c\*d\*sizeof\(input\) + a\*b\*c\*d\*sizeof\(BOOL\) + 12.54KB<UB
3. 当输入数据类型为DT\_INT16、DT\_UINT16、DT\_INT32、DT\_UINT32时，走整数计算路径，临时内存占用与输入数据类型相关：DT\_INT16/DT\_UINT16需4KB，DT\_INT32/DT\_UINT32需8KB。TileShape大小约束为：a\*b\*c\*d\*sizeof\(input\) + a\*b\*c\*d\*sizeof\(BOOL\) + sizeof\(input\)\*2KB<UB。
4. Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

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
a = pypto.tensor([5], pypto.DT_INT32)
out = pypto.logical_not(a)
```

结果示例如下：

```python
输入数据x: [0 1 2 3 4]
输出数据y: [True False False False False]
```
