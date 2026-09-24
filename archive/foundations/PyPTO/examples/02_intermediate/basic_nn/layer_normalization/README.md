# Layer Normalization 样例

本样例展示了如何使用 PyPTO 实现深度学习中常用的规范化层，包括标准 LayerNorm 和 RMSNorm。

## 总览介绍

规范化（Normalization）是 Transformer 架构中的关键组件。本样例涵盖了以下内容：
- **LayerNorm**: 带有均值（Mean）和方差（Variance）计算的标准层规范化。
- **RMSNorm**: 均方根规范化，是 LayerNorm 的一种简化变体，常用于 LLaMA 等大模型。
- **静态形状 (Static Shapes)**: 固定 Batch Size 的高效实现。
- **动态形状 (Dynamic Shapes)**: 支持可变 Batch Size 的实现方式。

## 代码结构

- **`layer_norm.py`**: 包含 LayerNorm 和 RMSNorm 的核心实现及其测试。

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
# 运行所有规范化样例
python3 layer_norm.py

# 列出所有可用的样例
python3 layer_norm.py --list
```

## 算法原理简述

### 1. LayerNorm
公式如下：
```
mean = mean(x, dim=-1)
var = var(x, dim=-1)
normalized = (x - mean) / sqrt(var + eps)
output = gamma * normalized + beta
```
具有可学习的缩放参数 `gamma` 和平移参数 `beta`。

### 2. RMSNorm
公式如下：
```
rms = sqrt(mean(x^2) + eps)
output = gamma * (x / rms)
```
仅包含缩放参数 `gamma`，计算开销更低。

## 关键技术点

### 动态形状支持
在处理可变 Batch 时，需要标记 `dynamic_axis` 并在 JIT 配置中开启支持：
```python
# 标记 Batch 维度为动态
x_pto = pypto.from_torch(x_torch, dynamic_axis=[0])

# 在 JIT 内核中配置
def layer_norm_dynamic(...):
    ...
```

### 归约运算
规范化层大量使用了 `pypto.sum` 算子来计算均值和平方和。

## 注意事项
- **精度**: 对于规范化运算，中间结果（如方差）的数值范围可能较大，建议在内部计算时使用 FP32 精度，最后输出再转回 FP16/BF16。
- **Tiling**: 建议根据 `hidden_size` 的大小来设置分块形状，以确保向量单元被充分利用。
- **验证**: 本样例所有输出均与 PyTorch 原生实现进行了精度比对。
