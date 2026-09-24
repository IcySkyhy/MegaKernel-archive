# 高级设计模式样例 (Advanced Design Patterns)

本目录展示了在开发大规模、工业级算子库时推荐使用的设计模式。

## 总览介绍

随着算子逻辑复杂度的增加，保持代码的可读性、可维护性和执行效率变得至关重要。本目录主要展示：
- **多函数协作 (Multi-Function Modules)**: 如何通过组合多个独立的 `pypto.frontend.jit` 函数来构建复杂的计算流水线，如完整的 Transformer Block。

## 样例代码特性

- **模块化构建**: 展示了如何像搭积木一样，将 LayerNorm, Linear 和 Activation 函数组合成一个大模块。
- **残差连接 (Residual Connection)**: 展示了在多函数调用路径中实现跳连的高效方式。
- **函数复用**: 展示了编译后的 JIT 函数如何在不同数据输入下被多次调用，而无需重新编译。

## 代码结构

- **`function/`**:
  - `function.py`: 展示了顺序串联、残差连接和函数复用等多种设计模式。
  - `multi_jit.py`: 展示了多 JIT 函数的创建调用模式。

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
cd function
python3 function.py
```

## 注意事项

- 在进行多函数组合时，注意输出张量的预分配，这在大模型推理中对减少内存碎片至关重要。
- PyPTO 后端会自动优化这些函数的组合执行，以尽可能减少不必要的内存读写。
