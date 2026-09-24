# 神经网络组件样例 (Neural Network Components)

本目录包含使用 PyPTO 构建常见神经网络组件的中级开发样例。这些组件是构建大型 Transformer 模型的基础。

## 总览介绍

在这一阶段，开发者将学习如何将基础算子组合成具有特定功能的神经网络层。本目录涵盖了以下核心组件：
- **Layer Normalization**: 展示了标准 LayerNorm 和 RMSNorm 的实现，涉及均值和方差的归约计算。
- **FFN Module (Feed-Forward Network)**: 实现完整的前馈网络，支持多种激活函数（ReLU, GELU, SwiGLU）以及动态 Batch Size 处理。

## 样例代码特性

本目录下的代码展示了 PyPTO 的以下高级应用特性：
- **模块化设计**: 演示如何通过封装 `pypto.frontend.jit` 函数来构建可复用的网络模块。
- **动态形状支持**: 在 FFN 模块中展示了如何使用 `dynamic_axis` 处理多变的 Batch 维度。
- **高性能算子组合**: 展示了矩阵乘法、逐元素运算与归约运算（如 Sum, Max）的深度集成。

## 代码结构

- **`layer_normalization/`**:
  - `layer_norm.py`: 包含 LayerNorm 和 RMSNorm 的核心实现及其精度验证。
- **`ffn/`**:
  - `ffn_module.py`: FFN 模块的核心实现与各种场景测试。

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

进入对应子目录后运行脚本：

```bash
# 运行 LayerNorm 样例
cd layer_normalization
python3 layer_norm.py

# 运行 FFN 样例
cd ../ffn
python3 ffn_module.py
```

## 学习建议

1. 建议先从 `layer_normalization` 开始学习，掌握基础的归约计算模式。
2. 随后学习 `ffn`，了解如何整合矩阵乘法与复杂的激活函数逻辑。
3. 参考各目录下的子 README 获取更详细的算法说明。
