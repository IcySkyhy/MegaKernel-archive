# 初级样例 (Beginner Examples)

本目录包含 PyPTO 的初级入门样例，旨在帮助开发者快速掌握 PyPTO 的核心概念和基础算子操作。

## 样例说明

初级样例分为以下几个类别：

### 1. 基础操作 ([basic](./basic))
- **内容**: 展示了最基础的张量创建、逐元素运算、矩阵乘法、归约运算、Tiling 配置以及变换操作（View + Assemble）。包含张量创建操作和符号标量（Symbolic Scalar）的使用。
- **核心特性**: `pypto.tensor`, `@pypto.frontend.jit`, `pypto.from_torch`, `pypto.view`, `pypto.assemble`, `pypto.scalar`。
- **推荐人群**: 初次接触 PyPTO 的开发者。

### 2. 计算算子 ([compute](./compute))
- **内容**: 详尽展示了各种计算算子的用法。
  - `elementwise_ops.py`: 逐元素算子（Add, Sub, Mul, Div, Exp, Log, Abs, Sqrt, Rsqrt, Clip 等），包含广播机制和标量运算。
  - `matmul_ops.py`: 矩阵乘法（Matmul）的各种配置。
  - `reduce_ops.py`: 归约算子（Sum, Max, Min 等）。
- **核心特性**: 各类数学计算 API。

### 3. Tiling (分块) 策略 ([tiling](./tiling))
- **内容**: 介绍如何配置硬件相关的 Tiling 形状，以优化在昇腾 NPU 上的执行效率。
- **核心特性**: `pypto.set_vec_tile_shapes`, `pypto.set_cube_tile_shapes`。

### 4. 变换算子 ([transform](./transform))
- **内容**: 展示张量形状变换、切片、转置等操作。
- **核心特性**: `pypto.transpose`, `pypto.reshape`, `pypto.slice`。

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

进入对应子目录运行脚本即可，例如：

```bash
cd basic
python3 basic_ops.py
```

## 学习建议

1. 首先阅读并运行 `basic/basic_ops.py`，了解 PyPTO 的基本工作流。
2. 根据需要深入学习 `compute` 下的各类算子 API。
3. 学习 `tiling` 了解如何根据硬件特性优化算子性能。
