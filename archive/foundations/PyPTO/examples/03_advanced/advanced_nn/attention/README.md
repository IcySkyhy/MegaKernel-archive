# Attention (注意力) 机制样例

本样例展示了如何使用 PyPTO 实现缩放点积注意力（Scaled Dot-Product Attention）机制，这是 Transformer 架构中最核心的组件。

## 总览介绍

本样例涵盖了从基础注意力计算到完整的多头注意力（Multi-head Attention）实现的全部过程：
- **缩放点积注意力**: 实现 `softmax(Q @ K^T / sqrt(d_k)) @ V`。
- **Q, K, V 投影**: 展示如何从隐藏状态投影得到查询、键和值。
- **多头拆分与拼接**: 展示张量的 `reshape` 和 `transpose` 操作以支持多头并行计算。
- **动态形状支持**: 支持可变的 Batch Size 和序列长度。
- **完整 Attention Block**: 包含输入输出投影的完整实现。

## 代码结构

- **`attention.py`**: 包含注意力机制的核心实现逻辑、配置类以及详细的测试用例。

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
# 运行所有注意力机制样例
python3 attention.py

# 列出所有可用的样例
python3 attention.py --list
```

## 核心算法实现

### 缩放点积注意力

```python
@pypto.frontend.jit
def scaled_dot_product_attention(
    q: pypto.tensor((S1, DQK), pypto.DT_FP32),
    k: pypto.tensor((S2, DQK), pypto.DT_FP32),
    v: pypto.tensor((S2, DV), pypto.DT_FP32),
    output: pypto.tensor((S1, DV), pypto.DT_FP32)
):
    # 1. 计算 Q @ K^T
    k_t = pypto.transpose(k, [0, 1, 3, 2])
    scores = pypto.matmul(q, k_t)

    # 2. 缩放与 Softmax
    scores_scaled = scores * scale
    attn_weights = pypto.softmax(scores_scaled, dim=-1)

    # 3. 施加到 V 上
    output[:] = pypto.matmul(attn_weights, v)
```

## 关键技术点

- **高效转置**: 在 NPU 上，通过 `pypto.transpose` 实现多维张量的高效重排。
- **分块策略 (Tiling)**: 针对注意力机制中的大规模矩阵乘法，配置最佳的 `cube_tile_shapes`。
- **动态轴标记**: 使用 `dynamic_axis=[0, 2]`（Batch 和 SeqLen）来应对推理时波动的输入长度。

## 最佳实践

- **数值稳定性**: 在计算 Softmax 之前进行缩放，防止指数爆炸。
- **内存布局**: 尽量保持 K 和 V 在内存中的连续性，以优化读取速度。
- **数据类型**: 在处理大模型注意力时，推荐使用 BF16 精度。

## 注意事项

- 注意力机制的内存复杂度为 O(N^2)，对于极长序列，请注意显存占用。
- 所有的实现均通过精度验证，确保与 PyTorch 的 `scaled_dot_product_attention` 结果一致。
