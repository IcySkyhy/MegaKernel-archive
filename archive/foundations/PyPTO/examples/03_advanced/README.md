# 高级样例 (Advanced Examples)

本目录包含 PyPTO 的高级开发样例，展示了复杂的架构实现、高级设计模式以及系统级的性能调优。

## 样例说明

高级样例涵盖以下核心领域：

### 1. 复杂神经网络架构 ([advanced_nn](./advanced_nn))
- **Attention Mechanism ([attention](./advanced_nn/attention))**:
    - 实现缩放点积注意力（Scaled Dot-Product Attention）。
    - 支持多头注意力（Multi-head Attention）。
    - 支持动态 Batch 和动态序列长度（Sequence Length）。
    - 展示了复杂的矩阵转置与乘法组合。

### 2. 设计模式 ([patterns](./patterns))
- **Multi-Function Module ([function](patterns/function))**:
    - 展示如何组合多个独立的 JIT 函数。
    - 函数间的顺序组合、残差连接（Residual Connection）。
    - 构建完整的 Transformer Block。

### 3. 系统级调优 ([cost_model](./cost_model))
- **Cost Model ([cost_model.py](./cost_model/cost_model.py))**:
    - 演示如何使用成本模型来评估和优化算子的执行效率。

### 4. 图捕获模式 ([aclgraph](./aclgraph))
- **ACLGraph ([aclgraph.py](./aclgraph/aclgraph.py))**:
   - 演示如何使用图捕获模式优化 Host 侧开销。

## 核心特性

在高级样例中，您将接触到：
- **复杂张量变换**: 频繁的 `transpose` 和 `reshape` 在注意力机制中的应用。
- **多函数协作**: 理解如何在大规模模型开发中通过组合多个小函数来保持代码的可读性和可维护性。
- **极致性能优化**: 通过对 Tiling、循环展开和硬件单元的深度适配来压榨硬件性能。

## 运行方法

在运行任何样例之前，请确保已配置 CANN 环境并设置了设备 ID：

```bash
# 配置 CANN 环境变量
# 安装完成后请配置环境变量，请用户根据set_env.sh的实际路径执行如下命令。
# 上述环境变量配置只在当前窗口生效，用户可以按需将以上命令写入环境变量配置文件（如.bashrc文件）。

# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置设备 ID
export TILE_FWK_DEVICE_ID=0
```

进入对应子目录运行脚本。

## 学习建议

1. 首先深入学习 `advanced_nn/attention`，这是所有现代 LLM 的核心。
2. 通过 `patterns/function` 学习如何组织大型算子项目。
3. 参考 `models` 目录下的真实模型实现，将这些高级特性应用到实际生产中。
