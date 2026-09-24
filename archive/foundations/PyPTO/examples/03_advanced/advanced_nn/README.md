# 高级神经网络架构样例 (Advanced NN Architectures)

本目录包含使用 PyPTO 实现的高级神经网络架构样例，重点关注大模型（LLM）中最核心的计算组件。

## 总览介绍

在这一阶段，开发者将面临更复杂的计算图和更严格的性能要求。本目录目前聚焦于：
- **Attention Mechanism**: 实现了缩放点积注意力（Scaled Dot-Product Attention）和完整的多头注意力（Multi-head Attention）模块。

## 样例代码特性

- **复杂张量变换**: 展示了如何通过频繁的 `transpose` 和 `reshape` 操作来实现多头注意力的拆分与合并。
- **极致性能优化**: 展示了针对注意力机制中的大规模矩阵乘法如何配置最优的 `cube_tile_shapes`。
- **动态 Batch 与序列长度**: 展示了在处理实时推理请求时，如何通过动态轴标记应对多变的输入规模。

## 代码结构

- **`attention/`**:
  - `attention.py`: 包含注意力机制的完整实现、配置类以及与 PyTorch 原生算子的精度比对。

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
cd attention
python3 attention.py
```

## 学习建议

1. 注意力机制是现代深度学习的基石，建议深入阅读 `attention/README.md` 了解算法细节。
2. 尝试修改 `attention.py` 中的 Tiling 配置，观察其对运行性能的影响。
3. 参考 `models` 目录下的真实模型代码，了解如何将本目录的组件应用到工业级项目中。
