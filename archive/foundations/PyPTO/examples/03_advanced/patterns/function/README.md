# 多函数模块组合样例 (Multi-Function Module)

本样例展示了如何将多个 `@pypto.frontend.jit` 函数组合在一起，构建复杂的计算流水线和模块化的神经网络架构。

## 总览介绍

在开发大型算子（如完整的 Transformer 层）时，将所有逻辑写在一个巨大的函数中是不利于维护和优化的。本样例演示了以下核心模式：
- **顺序组合 (Sequential Composition)**: 将多个 JIT 函数按顺序串联执行。
- **残差连接 (Residual Connection)**: 在不同函数执行路径间建立跳连。
- **函数复用 (Function Reuse)**: 使用不同的输入多次调用同一个 JIT 函数。
- **构建复杂模块**: 从小的功能函数（LayerNorm, Linear, Activation）逐步构建出完整的 Transformer Block。

## 代码结构

- **`function.py`**: 核心示例代码，展示了各种组合模式。
  - `test_sequential_functions`: 顺序执行示例。
  - `test_residual_connection`: 残差连接示例。
  - `test_transformer_block`: 综合示例，构建完整的 Transformer 块。
  - `test_function_reuse`: 函数复用示例。
- **`multi_jit.py`**: 多 JIT 函数组合创建示例，展示如何将多个独立编译的 JIT 函数串联使用。

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
# 运行所有组合模式样例
python3 function.py

# 列出所有可用的样例
python3 function.py --list
```

## 核心模式解析

### 1. 顺序串联
```python
# 第一步：标准化
layer_norm([x, gamma, beta], [normed])
# 第二步：激活
gelu_activation([normed], [activated])
```

### 2. 残差连接
```python
# 1. 正常分支计算
processed = process(x)
# 2. 残差相加
residual_add([x, processed], [output])
```

### 3. 函数复用
JIT 函数在第一次调用时编译，之后的重复调用将直接运行编译好的二进制代码，即使输入张量的具体数值不同。

## 最佳实践
- **职责单一**: 每个 JIT 函数应专注于完成一个独立的功能。
- **预分配内存**: 在调用函数前，提前准备好输出张量，避免在循环中重复申请显存。
- **模块化设计**: 模仿 PyTorch 的 `nn.Module` 风格，将常用的计算逻辑封装成可复用的功能函数。

## 注意事项
- 跨函数传递数据时，确保数据类型（Dtype）和设备（Device）保持一致。
- 函数组合的性能通常与算子融合（Operator Fusion）策略有关，PyPTO 后端会尝试优化这些组合。
