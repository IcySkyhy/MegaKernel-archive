# Tiling (分块) 操作样例

本目录包含 PyPTO 中 Tiling 配置操作的使用示例，涵盖了 Cube Tiling 和 Vector Tiling。

## 总览介绍

Tiling 是昇腾 NPU 性能优化的核心。通过合理地划分张量分块，可以最大化硬件单元（如 Cube 单元和 Vector 单元）的并行利用率，并优化内存访问模式。

样例涵盖以下内容：
- **Cube Tile Shapes**: 专为矩阵乘法（Cube 运算）设计的 Tiling 配置。
- **Vector Tile Shapes**: 专为逐元素运算和向量运算设计的 Tiling 配置。

## 代码文件说明

- **`tiling_config.py`**: 包含所有 Tiling 配置相关的示例。
  - `test_set_cube_tile_shapes_basic`: Cube Tiling 基础用法。
  - `test_set_vec_tile_shapes_basic`: Vector Tiling 基础用法。
  - `test_different_tile_shapes_on_results`: 验证不同 Tiling 形状下计算结果的一致性。
  - `test_different_tile_shapes_on_runtime`: 展示不同 Tiling 配置对运行性能的影响。

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

### 运行样例

```bash
# 运行所有 Tiling 相关的示例
python3 tiling_config.py

# 列出所有可用的 Tiling 用例
python3 tiling_config.py --list

# 运行特定的用例
python3 tiling_config.py cube_tile::test_set_cube_tile_shapes_basic
```

## 核心 API 与概念

### 1. Cube Tiling (矩阵乘法)
Cube Tiling 配置三个维度的分块大小：M、K 和 N。
```python
# 设置 Cube Tile 形状：[M_tile], [K_tile], [N_tile]
pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
```

### 2. Vector Tiling (向量/逐元素运算)
Vector Tiling 的分块数量必须与张量的维度（1-4维）匹配。
```python
# 设置 3 维张量的 Vector Tile 形状
pypto.set_vec_tile_shapes(1, 2, 8)
```

### 3. Tiling 对性能的影响
- **一致性**: 无论如何划分 Tiling，最终的计算结果应当保持一致。
- **性能**: 合理的 Tiling 形状可以显著减少 L1/L0 缓存与 Global Memory 之间的数据搬运次数，提升计算单元的利用率。

## 最佳实践
- **匹配算子类型**: 矩阵运算使用 Cube Tiling，向量运算使用 Vector Tiling。
- **对齐硬件**: 昇腾 NPU 硬件通常有特定的对齐要求（如 16x16 或 32x32），Tiling 形状建议参考硬件架构规格。
- **动态调整**: 在开发复杂算子时，可以通过实验不同的 Tiling 组合来找到性能最优解。

## 注意事项
- Tiling 配置必须在 JIT 内核函数内部、实际计算发生之前进行。
- 验证分块大小时，请确保分块形状不大于张量的实际形状（除非开启了自动 Padding 机制）。
