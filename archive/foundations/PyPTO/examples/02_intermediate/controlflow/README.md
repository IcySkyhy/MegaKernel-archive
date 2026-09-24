# 运行时高级特性样例 (Runtime Features)

本目录展示了 PyPTO 运行时（Runtime）的各项高级特性，帮助开发者处理真实世界中复杂的执行逻辑。

## 总览介绍

除了基础的算子计算，实际应用中常需要处理动态形状、条件判断和循环控制。本目录涵盖了以下内容：
- **动态形状 (Dynamic Shapes)**: 演示如何处理推理过程中波动的 Batch Size 或序列长度。
- **控制流 (Control Flow)**: 包含算子内部的条件分支和复杂的循环结构 (`pypto.loop`)。
- **输入顺序灵活性**: 演示 JIT 内核对输入参数顺序的容错能力。

## 样例代码特性

- **`others/dynamic.py`**: 展示 `dynamic_axis` 的标记方法以及在 JIT 配置中开启对非对齐动态形状的支持。
- **`condition/condition.py`**: 展示如何在内核函数中使用 `if_else` 构建复杂的逻辑分支。
- **`loop/`**: 深入介绍 `pypto.loop` 的高级用法，如循环展开（Unroll）和编译期打印调试。
- **`others/kernel_input.py`**: 展示 JIT 内核对输入参数顺序的容错能力。

## 代码结构

- **`condition/`**: 条件分支示例目录，详见 [condition/README.md](condition/README.md)。
- **`loop/`**: 循环控制专题目录，详见 [loop/README.md](loop/README.md)。
- **`others/`**: 其他运行时特性示例目录，详见 [others/README.md](others/README.md)。
  - `dynamic.py`: 动态形状处理示例。
  - `kernel_input.py`: 输入顺序灵活性示例。

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

直接运行对应的 Python 脚本：

```bash
python3 others/dynamic.py
python3 condition/condition.py
python3 others/kernel_input.py
```

## 注意事项

- 内核中的循环和条件判断直接影响代码生成，建议参考 `loop/README.md` 了解更多底层细节。
