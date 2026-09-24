# Softmax 算子开发样例

本样例展示了如何使用 PyPTO 框架从底层逻辑开始实现一个高效的 Softmax 算子。

## 总览介绍

Softmax 是深度学习中极其常用的算子，尤其在 Attention 机制中。本样例详细介绍了以下实现步骤：
- **数值稳定的计算策略**: 通过减去最大值（Max Subtraction）防止指数运算溢出。
- **动态形状支持**: 处理不固定的 Batch Size。
- **显式分块 (Tiling) 配置**: 优化 NPU 硬件执行效率。
- **内核循环处理**: 在内核中通过循环遍历不同的数据分块。

## 核心算法实现

### 1. 核心计算逻辑 (`softmax_core`)

```python
def softmax_core(x: pypto.Tensor) -> pypto.Tensor:
    # 找到最后维度的最大值
    row_max = pypto.amax(x, dim=-1, keepdim=True)
    # 减去最大值，保证数值稳定性
    sub = x - row_max
    # 计算指数
    exp = pypto.exp(sub)
    # 计算指数之和
    esum = pypto.sum(exp, dim=-1, keepdim=True)
    # 归一化
    return exp / esum
```

### 2. JIT 内核封装 (`softmax_kernel`)

内核函数负责管理 Tiling 和循环：

```python
@pypto.frontend.jit
def softmax_kernel(
    x: pypto.Tensor(x_shape, pypto.DT_FP32),
):
    # 设置 Tiling 形状
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    # 使用 pypto.loop 处理数据分块
    for idx in pypto.loop(b_loop):
        # ... 视图划分与计算 ...
```

## 代码文件说明

- **`softmax.py`**: 包含完整的 Softmax 实现代码及测试验证逻辑。

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

### 执行脚本

```bash
python3 softmax.py
```

## 关键特性

- **数值稳定性**: 通过 `amax` 和 `sub` 组合实现，这是在 NPU 上开发高性能激活算子的标准实践。
- **与 PyTorch 验证**: 实现结果通过 `assert_allclose` 与 `torch.softmax` 进行比对。
- **高性能 Tiling**: 展示了如何针对向量处理单元（Vector Core）配置最佳的计算分块。

## 注意事项

- 算子的性能高度依赖于 `set_vec_tile_shapes` 的设置，建议根据实际的隐层维度（Hidden Size）进行调优。
- 本样例展示的是在 dim=-1 上的 Softmax，如需在其他维度计算，需相应调整 `amax` 和 `sum` 的维度参数。
