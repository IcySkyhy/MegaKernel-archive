# 基础操作样例 (Basic Operations)

本样例展示了 PyPTO 的基础算子操作和典型的编程模式，非常适合初学者快速上手。

## 样例代码特性

- **JIT 编译**: 使用 `@pypto.jit` 装饰器定义可以在 NPU 上执行的内核函数。
- **PyTorch 集成**: 能够直接接受 PyTorch NPU 张量作为输入输出，无缝衔接现有深度学习工作流。
- **显式 Tiling 控制**: 通过 `set_vec_tile_shapes` 和 `set_cube_tile_shapes` 手动优化硬件执行效率。
- **动态Shape**: 支持在运行时动态调整张量的形状，无需在编译时就确定。

## 代码结构

- **`basic_ops.py`**: 包含所有示例的主脚本，作为快速上手的总览入口。
  - `test_add()`: 示例 1 - 加法运算。
  - `test_erfc()`: 示例 2 - 逐元素运算（ERFC）。
  - `test_matmul()`: 示例 3 - 矩阵乘法。
  - `test_sum()`: 示例 4 - 归约运算（Sum）。
  - `test_dynamic_add()`: 示例 5 - 动态Shape

更详细的各类算子用法，请参考同级目录：

- `../compute/`: 逐元素算子、矩阵乘法、归约算子。
- `../tiling/`: Tiling 配置策略。
- `../transform/`: 变换算子。

## 运行方法

### 环境准备

```bash
# 配置 CANN 环境变量
# 安装完成后请配置环境变量，请用户根据set_env.sh的实际路径执行如下命令。
# 上述环境变量配置只在当前窗口生效，用户可以按需将以上命令写入环境变量配置文件（如.bashrc文件）。

# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 执行脚本

```bash
# 运行所有示例
python3 basic_ops.py

# 运行特定示例（例如示例 1：加法运算）
python3 basic_ops.py -t add
```

## 注意事项

- **分块大小**: Tiling 形状的选择会显著影响算子性能，通常应根据 NPU 架构的向量/矩阵计算单元大小来设定。
- **环境**: 确保 `torch_npu` 已正确安装并能识别到昇腾显卡。
