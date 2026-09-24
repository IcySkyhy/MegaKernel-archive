# 循环 (Loop) 特性样例

本样例展示了 PyPTO 中循环控制的高级特性及其在算子内核中的使用规则。

## 总览介绍

在 PyPTO 的 JIT 内核中，循环不仅用于遍历数据，还涉及到编译期的优化和代码生成。本样例涵盖了以下关键点：
- **基础循环用法**: 包括 `start`, `end`, `step` 参数的使用。
- **循环起始与结束判定**: 如何判断循环的开始和结束。
- **循环展开 (Unroll)**: 使用 `unroll` 接口优化性能。
- **算子位置规则**: 算子通常必须放置在最内层循环中以确保正确生成代码。
- **编译期打印特性**: 了解内核循环中的 `print` 行为（仅在编译期执行）。

## 代码结构

- **`loop.py`**: 包含循环特性的详细示例和验证逻辑。
  - `test_loop_basic`: 基础循环用法（start/stop/step）。
  - `test_loop_compile_phase_print`: 编译期打印特性。
  - `test_add_scalar_loop`: 标量加法循环示例。
  - `test_add_scalar_loop_dyn_axis`: 动态轴下的标量加法循环示例。

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
# 运行所有循环特性示例
python3 loop.py

# 列出所有可用的用例
python3 loop.py --list
```

## 核心概念与规则

### 1. 基础用法
```python
for i in pypto.loop(start=0, end=10, step=1):
    # 循环体逻辑
```

### 2. 算子放置规则 (OP Position Rule)
为了确保计算指令能够被正确地分块（Tiling）并生成高性能的 NPU 代码，计算类算子（如 `add`, `matmul` 等）应当放置在内核的最内层循环中。

### 3. 编译期打印
在 `pypto.loop` 内部直接使用 `print()` 时，打印行为发生在编译阶段，而不是运行阶段。这意味着它不能打印运行时的张量数值，只能用于辅助调试编译路径。

### 4. 循环展开
对于迭代次数较少的循环，可以使用展开技术来减少分支开销。

## 注意事项
- 循环的步长 `step` 必须是常数。
- 内核中的循环结构直接影响到 Tiling 策略的执行，复杂的循环嵌套可能需要更精细的分块配置。
