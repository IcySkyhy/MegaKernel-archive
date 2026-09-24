# 自定义激活函数样例 (Custom Activation)

本样例展示了如何通过组合 PyPTO 的基础算子来实现自定义的、复杂的激活函数。

## 总览介绍

在现代 Transformer 架构（如 LLaMA, GPT, PaLM）中，经常会用到一些非标准的激活函数。本样例演示了以下激活函数的 PyPTO 实现：
- **SiLU (Swish)**: `x * sigmoid(x)`。
- **GELU**: 高斯误差线性单元的近似实现。
- **SwiGLU**: `Swish(gate) * up`，门控线性单元的一种。
- **GeGLU**: `GELU(gate) * up`。

## 代码结构

- **`activation.py`**: 包含各种自定义激活函数的实现逻辑及其与 PyTorch 原生算子的对比验证。

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
# 运行所有激活函数样例
python3 activation.py

# 列出所有可用的样例
python3 activation.py --list
```

## 实现模式：算子组合 (Composition)

PyPTO 允许开发者像写数学公式一样组合算子。以 **SiLU** 为例：

```python
def silu_activation(x: pypto.Tensor) -> pypto.Tensor:
    # 1. 计算 sigmoid(x) = 1 / (1 + exp(-x))
    x_neg = pypto.mul(x, -1.0)
    exp_neg = pypto.exp(x_neg)
    sigmoid = pypto.div(1.0, pypto.add(exp_neg, 1.0))

    # 2. 计算 x * sigmoid(x)
    return pypto.mul(x, sigmoid)
```

## 关键技术点

- **无缝集成**: 自定义激活函数可以像内置算子一样在 `@pypto.frontend.jit` 内核中使用。
- **门控机制 (Gating)**: 展示了处理双输入（Gate 和 Up）算子的典型模式，这在现代大模型中非常常见。
- **精度验证**: 所有算子均通过 `assert_allclose` 与 PyTorch 的标准实现进行比对。

## 注意事项
- **算子融合**: PyPTO 会在后端尽可能地融合这些组合算子，以减少不必要的内存读写。
- **数据类型**: 建议在 BF16 精度下进行开发，以保证精度并获得硬件加速。
- **分块 (Tiling)**: 激活函数通常是逐元素运算，分块形状应尽可能填满向量处理单元。
