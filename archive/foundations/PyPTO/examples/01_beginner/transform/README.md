# 变换算子样例 (Transform Operations)

本目录包含 PyPTO 中张量变换算子的使用示例，包括 Assemble, Gather, Concat 和 View 等操作。

## 总览介绍

变换算子样例涵盖以下内容：
- **Assemble**: 将一个小张量放置到大张量的指定偏移位置。
- **Gather**: 根据索引张量从原张量中收集元素。
- **Concat**: 沿指定维度拼接多个张量。
- **View**: 创建张量的视图，支持指定形状和偏移，不涉及数据拷贝。

## 代码文件说明

- **`transform_ops.py`**: 包含所有变换算子的综合示例。
  - `test_assemble_basic`: 基础 Assemble 操作。
  - `test_gather_basic`: 基础 Gather 操作。
  - `test_concat_basic`: 基础 Concat 操作。
  - `test_view_basic`: 基础 View 操作。
- **`add_scalar_loop_view_assemble.py`**: 标量加法结合 View 与 Assemble 的循环分块示例。

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
# 运行所有变换相关的示例
python3 transform_ops.py

# 列出所有可用的变换用例
python3 transform_ops.py --list

# 运行特定的用例
python3 transform_ops.py assemble::test_assemble_basic
```

## 核心 API 说明

### 1. Assemble
常用于将计算后的分块（Tile）结果拼回到全局张量中。
```python
# 将 small_tensor 放置在 large_tensor 的 [0, 0] 偏移处
pypto.assemble(small_tensor, offsets=[0, 0], large_tensor)
```

### 2. Gather
根据索引从输入张量中选取数据。
```python
# 沿维度 0 进行 Gather
pypto.gather(input_tensor, dim=0, index_tensor)
```

### 3. Concat
将多个张量按顺序拼接。
```python
# 沿维度 1 拼接两个张量
pypto.concat([tensor1, tensor2], dim=1)
```

### 4. View
创建指向原张量部分区域的引用，是实现 Tiling 循环的关键。
```python
# 创建一个 4x4 的视图，偏移量为 [0, 4]
view = pypto.view(tensor, shape=[4, 4], offsets=[0, 4])
```

## 注意事项
- **Assemble 与 View**: 这两个算子通常配合使用来实现手动的 Tiling 循环。
- **维度对齐**: 在进行 Concat 操作时，除了拼接维度外，其他维度的形状必须一致。
- **数据拷贝**: `view` 操作不涉及数据拷贝，而 `gather` 和 `concat` 通常会产生新的数据副本。
