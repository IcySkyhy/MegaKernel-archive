# 自定义算子开发样例 (Custom Operator Operations)

本目录包含使用 PyPTO 开发自定义算子的中级样例，涵盖了算子组合与高性能算子实现的最佳实践。

## 总览介绍

自定义算子开发是 PyPTO 的核心能力之一。通过本目录的样例，您将学习到：
- **算子组合 (Composition)**: 如何利用基础数学算子构建复杂的非标准激活函数。
- **高性能实现**: 以 Softmax 为例，展示如何通过数值稳定算法和显式分块（Tiling）实现工业级性能。

## 样例代码特性

- **数值稳定性**: 展示了在 NPU 上处理指数运算时减去最大值的经典策略。
- **高度可定制**: 展示了如何通过组合 `exp`, `sum`, `amax` 等基础算子实现 SiLU, GELU, SwiGLU 等主流模型算子。
- **与 PyTorch 集成**: 所有自定义算子均提供了精度对比逻辑，确保与标准框架行为一致。

## 代码结构

- **`activation/`**:
  - `activation.py`: 实现了 SiLU, GELU, SwiGLU, GeGLU 等多种复杂激活函数。
- **`softmax/`**:
  - `softmax.py`: 详细展示了 Softmax 算子的分步实现与优化。

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

进入对应子目录后运行：

```bash
# 运行激活函数样例
cd activation
python3 activation.py

# 运行 Softmax 样例
cd ../softmax
python3 softmax.py
```

## 注意事项

- 在开发涉及非线性变换（如 Exp, Log）的算子时，务必注意数据溢出问题。
- 自定义算子的性能受 Tiling 策略影响较大，建议根据实际业务场景的 Shape 进行调优。
