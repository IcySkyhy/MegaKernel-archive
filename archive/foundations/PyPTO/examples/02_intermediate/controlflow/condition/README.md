# 条件分支样例 (Condition)

本样例展示了如何在 PyPTO 的 JIT 内核中使用条件分支逻辑。

## 总览介绍

在实际算子开发中，常需要根据不同条件执行不同的计算路径。本样例涵盖了以下场景：
- **嵌套循环中的条件判断**: 在多层循环内使用 `if_else` 构建条件分支。
- **动态轴 + 静态条件**: 使用编译期布尔标志（`bool` flag）控制分支。
- **动态轴 + 动态条件**: 使用运行时索引比较（`index comparison`）控制分支。
- **动态轴 + 循环边界条件**: 使用 `is_loop_begin` / `is_loop_end` 进行边界处理。

## 代码结构

- **`condition.py`**: 包含所有条件分支示例的整合脚本。
  - `test_nested_loops_with_conditions`: 嵌套循环中的条件判断。
  - `test_add_scalar_loop_dyn_axis_static_cond`: 动态轴 + 静态条件。
  - `test_add_scalar_loop_dynamic_axis_dynamic_cond`: 动态轴 + 动态条件。
  - `test_add_scalar_loop_dynamic_axis_dynamic_loop_cond`: 动态轴 + 循环边界条件。

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
# 运行所有条件分支示例
python3 condition.py

# 列出所有可用的用例
python3 condition.py --list

# 运行特定的用例
python3 condition.py nested_loops_with_conditions::test_nested_loops_with_conditions
```

## 注意事项

- 条件分支会影响编译器的代码生成路径，复杂的嵌套条件可能导致编译时间增加。
- 动态条件与静态条件的选择取决于条件值是否在编译期可知。
