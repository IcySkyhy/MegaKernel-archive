# 计算算子样例 (Compute Operations)

本目录包含 PyPTO 中各种计算算子的使用示例，包括逐元素运算、矩阵乘法和归约运算。

## 总览介绍

计算算子样例涵盖以下内容：
- **逐元素运算 (Element-wise Operations)**: 对张量的每个元素独立应用的算术运算，如加、减、乘、除、绝对值、指数、对数等。
- **矩阵乘法 (Matrix Multiplication)**: 各种配置下的矩阵乘法运算。
- **规约运算 (Reduction Operations)**: 沿指定维度对张量进行缩减的运算，如求和、最大值、最小值等。

## 代码文件说明

- **`elementwise_ops.py`**: 逐元素算子示例，涵盖 `abs`, `add`, `clip`, `div`, `exp`, `expm1`, `log`, `mul`, `neg`, `pow`, `rsqrt`, `sqrt`, `ceil`, `floor`, `trunc`, `round`, `sub`。
- **`matmul_ops.py`**: 矩阵乘法示例，涵盖基础矩阵乘法、批量矩阵乘法、广播矩阵乘法、带转置的矩阵乘法以及带 Bias 的矩阵乘法。
- **`reduce_ops.py`**: 归约算子示例，涵盖 `amax`, `amin`, `maximum`, `minimum`, `sum`。

## 运行方法

### 环境准备

```bash
# 配置 CANN 环境变量
# 安装完成后请配置环境变量，请用户根据set_env.sh的实际路径执行如下命令。
# 上述环境变量配置只在当前窗口生效，用户可以按需将以上命令写入环境变量配置文件（如.bashrc文件）。

# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置设备 ID
export TILE_FWK_DEVICE_ID=0
```

### 运行样例

每个脚本都支持列出所有用例或运行特定用例：

```bash
# 运行所有逐元素运算样例
python3 elementwise_ops.py

# 列出所有可用的逐元素运算用例
python3 elementwise_ops.py --list

# 运行特定的用例
python3 elementwise_ops.py abs::test_abs_basic
```

## 算子特性说明

### 逐元素运算

支持广播（Broadcasting）机制和标量（Scalar）操作。

```python
@pypto.frontend.jit()
def add_example(
    a: pypto.Tensor(shape, dtype),
    b: pypto.Tensor(shape, dtype),
    out: pypto.Tensor(shape, dtype)
):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.add(a, b)

```

### 矩阵乘法

使用 Cube Tiling 进行高效计算，支持指定输出数据类型。

```python
@pypto.frontend.jit()
def matmul_example(
    a: pypto.Tensor((M, K), pypto.DT_BF16),
    b: pypto.Tensor((K, N), pypto.DT_BF16),
    out: pypto.Tensor((M, N), pypto.DT_BF16)
):
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    out[:] = pypto.matmul(a, b, out_dtype=pypto.DT_BF16)
```

### 归约运算

支持指定维度（dim）和是否保持维度（keepdim）。

```python
@pypto.frontend.jit()
def sum_example(
    x: pypto.Tensor(x_shape, dtype),
    out: pypto.Tensor(out_shape, dtype)
):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.sum(x, dim=0, keepdim=True)
```

## 注意事项

- 在进行矩阵乘法时，建议显式设置 Cube Tile 形状以获得最佳性能。
- 归约操作通常涉及到跨 Tile 的数据交互，请注意 Tiling 的划分策略。
- 所有的样例都包含与 PyTorch 原生算子的对比验证，确保计算结果的准确性。
