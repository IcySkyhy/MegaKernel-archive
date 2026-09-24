# pypto.Tensor索引功能说明

Tensor索引是Tensor的核心操作之一，用于从Tensor中筛选、提取或修改特定位置的元素。通过索引操作，开发者可精准获取Tensor中的部分数据（如单个元素、子Tensor、特定维度数据），或对指定位置元素进行赋值修改。

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

## 一、\_\_getitem\_\_

## 功能说明

通过索引或切片的方式从Tensor中获取子Tensor或单个元素等，该方法支持多种索引模式，提供了灵活且直观的数据访问方式。

## 函数原型

```python
def __getitem__(self, key, *, valid_shape: Optional[List[Union[int, SymbolicScalar]]] = None)
```

## 参数说明

| 参数名      | 输入/输出 | 说明                                                                 |
|-------------|-----------|----------------------------------------------------------------------|
| key         | 输入      | Tensor索引，用于获取Tensor对应位置的数据。<br> 支持类型：<br> - int或SymbolicScalar（符号标量）: 单个整数索引。<br> - slice: 切片对象。<br> - tuple: 多维索引的组合，类型包括：int或SymbolicScalar，slice，Ellipsis(...)。 |
| valid_shape | 输入      | 表示输出Tensor有效数据的大小。 |

## 返回值说明

返回对应索引位置的Tensor数据。

## 约束说明

1.对于slice（切片对象，格式为start:end:step），当前功能暂时不支持step设置,值默认固定为1。

未支持示例：a\[1:2:2, :\]。

2.当前功能暂时不支持bool类型索引。

未支持示例：a\[True, False, True, False\]。

3.当前功能暂时不支持Tensor类型索引。

未支持示例：a\[b\]，\(b = pypto.Tensor\(\[2\], pypto.DT\_INT32\)。

## 使用示例

1. 全切片（slice）

   使用切片获取Tensor的子区域

   ```python
   a = pypto.tensor([4, 4], pypto.DT_FP32)
   b = a[:2, :2] #等价于view(a, [2, 2], [0, 0])
   ```

   结果示例如下：

   ```python
   输入数据a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   输出数据b: [[1, 2],
               [5, 6]]
   ```

2. 混合索引和切片

   结合整数索引和切片，可以降低维度并提取特定行或列。

   ```python
   a = pypto.tensor([4, 4], pypto.DT_FP32)
   b = a[1, 1:3] #等价于先view(a, [1, 2], [1, 1])，再reshape为 [2]
   ```

   结果示例如下：

   ```python
   输入数据a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   输出数据b: [6, 7]
   ```

3. 负数索引

    支持Python风格的负索引，从末尾开始计数。

    ```python
    a = pypto.tensor([4, 4], pypto.DT_FP32)
    b = a[-1, -3:-1] #等价于s[3, 1:3]
    ```

    结果示例如下：

    ```python
    输入数据a: [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]]
    输出数据b: [14, 15]
    ```

4. 省略号（...）

   使用 \`...\`自动填充中间的所有维度，简化多维索引。

   ```python
   a = pypto.tensor([4, 4], pypto.DT_FP32)
   b = a[..., 1:3] #等价于s[:, 1:3]
   ```

   结果示例如下：

   ```python
   输入数据a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   输出数据b: [[2, 3],
               [6, 7],
               [10, 11],
               [14, 15]]
   ```

5. 单元素访问

   整数索引，取出Tensor的单个元素（仅支持DT\_INT32类型）。

   ```python
   a = pypto.tensor([4, 4], pypto.DT_INT32)
   b = a[0, 0] #返回SymbolicScalar
   ```

   结果示例如下：

   ```python
   输入数据a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   输出数据b: 1
   ```

6. Gather操作

   当索引为\[int:Tensor\]的形式时，执行gather操作，索引中int类型对应dim，Tensor类型对应index，该切片语法等价于Tensor.gather\(dim, index\)。

   ```python
   a = pypto.tensor([4, 4], pypto.DT_FP32)
   index = pypto.tensor([1, 4], pypto.DT_INT32)
   b = a[0:index] #调用gather(a, 0, index)
   ```

   结果示例如下：

   ```python
   输入数据a: [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]]
   输入数据index: [[0, 1, 2, 3]]
   输出数据b: [[1, 6, 11, 16]]
   ```

## 二、\_\_setitem\_\_

## 功能说明

通过索引或切片的方式向Tensor的指定位置赋值。

## 函数原型

```python
def __setitem__(self, key, value)
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| key     | 输入      | Tensor索引，用于获取Tensor对应位置的数据。<br> 支持类型：<br> - int或SymbolicScalar（符号标量）: 单个整数索引。<br> - slice: 切片对象。<br> - tuple: 多维索引的组合，类型包括：int或SymbolicScalar，slice，Ellipsis(...)。 |
| value   | 输入      | 需要设置的值，类型支持Tensor或标量（float/int）。 |

## 返回值说明

返回对应位置赋值后的Tensor。

## 约束说明

1.对于slice（切片对象，格式为start:end:step），当前功能暂时不支持step设置,值默认固定为1。

未支持示例：a\[1:2:2, :\]。

2.当前功能暂时不支持bool类型索引。

未支持示例：a\[True, False, True, False\]。

3.当前功能暂时不支持Tensor类型索引。

未支持示例：a\[b\]，\(b = pypto.Tensor\(\[2\], pypto.DT\_INT32\)。

## 使用示例

1. 全切片（slice）

   使用切片将一个小Tensor组装到大Tensor的指定位置。

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_FP32)
   b = pypto.Tensor([2, 2], pypto.DT_FP32)
   a[0:, 0:] = b #等价于assemble(b, (0, 0), a)
   ```

   结果示例如下：

   ```python
   输入数据a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   输入数据b: [[10, 10]
               [10, 10]]
   输出数据a: [[10, 10, 0, 0],
               [10, 10, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   ```

2. 混合索引和切片

   结合整数索引和切片，可以对特定行或列进行操作。

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_FP32)
   b = pypto.Tensor([2], pypto.DT_FP32)
   a[0, 1:3] = b #b被reshape为(1, 2),等价于pypto.assemble(b, (0, 1), a)
   ```

   结果示例如下：

   ```python
   输入数据a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   输入数据b: [10, 10]
   输出数据a: [[0, 10, 10, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   ```

3. 负索引

   支持Python风格的负索引，从末尾开始计数。

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_FP32)
   b = pypto.Tensor([2], pypto.DT_FP32)
   a[-1, -3:-1] = b #等价于a[3, 1:3]
   ```

   结果示例如下：

   ```python
   输入数据a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   输入数据b: [10, 10]
   输出数据a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 10, 10, 0]]
   ```

4. 省略号（...）

   使用... 可以自动填充中间维度。

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_FP32)
   b = pypto.Tensor([2, 2], pypto.DT_FP32)
   a[..., 2:4] = b #等价于a[0:2, 2:4]
   ```

   结果示例如下：

   ```python
   输入数据a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   输入数据b: [[10, 10]
               [10, 10]]
   输出数据a: [[0, 0, 10, 10],
               [0, 0, 10, 10],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   ```

5. 单元素赋值

   整数索引，对单个元素赋值（仅支持DT\_INT32类型）。

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_INT32)
   a[2, 3] = 5 #调用SetTensorData
   ```

   结果示例如下：

   ```python
   输入数据a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   输出数据a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 5],
               [0, 0, 0, 0]]
   ```

6. Scatter操作

   当key为slice，key.start为int， key.stop为Tensor时\(a\[start:stop\]   \)，执行scatter操作。

   ```python
   a = pypto.Tensor([4, 4], pypto.DT_FP32)
   indices = pypto.Tensor([1, 4], pypto.DT_INT32) # 索引Tensor
   values = pypto.Tensor([1, 4], pypto.DT_FP32)
   # 在维度0上进行scatter
   a[0:indices] = values #调用pypto.scatter(a, 0, indices, values)
   ```

   结果示例如下：

   ```python
   输入数据a: [[0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]]
   输入数据indices：[[0, 1, 2, 3]]
   输入数据values：[[10, 10, 10, 10]]
   输出数据a: [[10, 0, 0, 0],
               [0, 10, 0, 0],
               [0, 0, 10, 0],
               [0, 0, 0, 10]]
   ```
