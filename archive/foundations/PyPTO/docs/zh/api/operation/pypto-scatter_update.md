# pypto.scatter\_update

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

功能1：原地操作，将4维src根据2维索引index更新到4维input上，计算公式如下：

$$
input\left[\frac{\text{index}[i][j]}{\text{blockSize}}\right]\left[\text{index}[i][j] \% \text{blockSize}\right][0][\dots] = src[i][j][0][\dots]
$$

功能2：原地操作，将2维src根据2维index更新到2维input上，计算公式如下（其中s是index第二维的大小，即index.shape[1]）：

$$
input[[\text{index}[i][j]][\dots]] = src[i*s + j][\dots]
$$

## 函数原型

```python
scatter_update(input: Tensor, dim: int, index: Tensor, src: Tensor) -> Tensor
```

## 参数说明

| 参数名  | 输入/输出 | 说明                                                                 |
|---------|-----------|----------------------------------------------------------------------|
| input   | 输入      | 支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32，DT_FP16，DT_BF16，DT_INT32，DT_INT16。<br>支持的维度：2维，4维<br>2维Shape[blockNum * blockSize, d]，4维Shape[blockNum, blockSize, 1, d]<br>不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |
| dim     | 输入      | 请保持默认值-2。 |
| index   | 输入      | input的一组索引。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_INT64，DT_INT32，DT_INT16。<br>支持的维度：2维<br>Shape[b, s] |
| src     | 输入      | src是一组更新值。<br>支持的类型为：Tensor。<br>Tensor支持的数据类型为：DT_FP32，DT_FP16，DT_BF16，DT_INT32，DT_INT16。数据类型和input保持一致<br>支持的维度：2维，4维<br>2维Shape[b * s, d]，4维Shape[b, s, 1, d]<br>不支持空Tensor；Shape Size不大于2147483647（即INT32_MAX）。 |

## 返回值说明

返回更新后的input，为inplace操作。

## 约束说明

broadcast约束：不支持broadcast。

Tensor格式约束：Tensor类型输入不支持`TileOpFormat.TILEOP_NZ`格式。

ViewShape约束：2维场景下ViewShape为\[viewB \* s, d\]，4维场景下ViewShape为\[viewB, viewS, 1, d\]，尾轴d不做切分。2维场景下，\[viewB \* s, d\]针对src做切分，其第0维是index的第1维s的倍数，\[viewB, S\]针对index做切分。4维场景下，\[viewB, viewS, 1, d\]针对src做切分，\[viewB, viewS\]针对index做切分。

TileShape约束：2维场景下TileShape为\[tileS, d\]，4维场景下TileShape为\[tileB, tileS, 1, d\]。尾轴d不做切分。2维场景下，TileShape针对src做切分，\[1,tileS\]针对index做切分，tileBS是输入index的第1维s的约数，如src为\[12, 64\]，index为\[3, 4\]，TileShape为\[TileS, 64\]，其中，TileS可以是1、2、4。4维场景下，TileShape针对src做切分，并且，\[tileB, tileS\]针对index做切分。由于TileShape的切分针对src和index，切块大小之和应小于UB限制。

二维示例：
input：[16, 8]，index：[5, 2]，src：[10, 8]，viewShape：[viewB \* s, 8]，viewB需要是整数，即第0维是s的倍数，tileShape：[tileS, 8]，tileS需要是s的约数即1或者2。

## 调用示例

### TileShape设置示例

调用该operation接口前，应通过set_vec_tile_shapes设置TileShape。

TileShape维度应和输入src一致。

输入input和输出均在gm上，不涉及tile切分。输入index和输入src需要搬入ub，涉及tile切分。

如输入input为[t, d]，dim为-2，输入index为[b, s]，输入src为[bs, d]，其中bs=b*s，输出为[t, d]，TileShape设置为[bs1, d1]，则bs1用于切分bs轴，d轴不允许切分，d1必须和d相等。

```python
pypto.set_vec_tile_shapes(16, 64)
```

### 接口调用示例

- 将2维src根据2维index更新到2维input上，注意原地操作的写法，即等号左边的输出应和输入input相同：

    ```python
    x = pypto.tensor([8, 3], pypto.DT_INT32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    z = pypto.tensor([4, 3], pypto.DT_INT32)
    x = pypto.scatter_update(x, -2, y, z)
    ```

    结果示例如下：

    ```python
    输入数据x:[[0 0 0],
               [0 0 0],
               [0 0 0],
               [0 0 0],
               [0 0 0],
               [0 0 0],
               [0 0 0],
               [0 0 0]]
    输入数据y:[[1 2],
               [4 5]]
    输入数据z:[[1 2 3],
               [4 5 6],
               [7 8 9],
               [10 11 12]]
    输出数据x:[[0 0 0],
               [1 2 3],
               [4 5 6],
               [0 0 0],
               [7 8 9],
               [10 11 12],
               [0 0 0],
               [0 0 0]]
    ```

- 将4维src根据2维索引index更新到4维input上，注意原地操作的写法，即等号左边的输出应和输入input相同：

    ```python
    x = pypto.tensor([2, 6, 1, 3], pypto.DT_INT32)
    y = pypto.tensor([2, 2], pypto.DT_INT64)
    z = pypto.tensor([2, 2, 1, 3], pypto.DT_INT32)
    x = pypto.scatter_update(x, -2, y, z)
    ```

    结果示例如下：

    ```python
    输入数据x:[[
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
               ],
               [
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[0 0 0]],
               ]]
    输入数据y:[[1 8],
               [4 10]]
    输入数据z:[[
                 [[1 2 3]],
                 [[4 5 6]],
               ],
               [
                 [[7 8 9]],
                 [[10 11 12]],
               ]]
    输出数据x:[[
                 [[0 0 0]],
                 [[1 2 3]],
                 [[0 0 0]],
                 [[0 0 0]],
                 [[7 8 9]],
                 [[0 0 0]],
               ],
               [
                 [[0 0 0]],
                 [[0 0 0]],
                 [[4 5 6]],
                 [[0 0 0]],
                 [[10 11 12]],
                 [[0 0 0]],
               ]]
    ```
