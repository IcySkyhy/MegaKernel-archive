# FFN 模块样例 (Feed-Forward Network)

本样例展示了如何使用 PyPTO 实现一个完整的前馈网络（FFN）模块。该模块设计用于昇腾 NPU 硬件的高效执行，支持多种激活函数和动态形状。

## 核心特性

- **多种激活函数**: 支持 GELU, SwiGLU 和 ReLU 激活。
- **静态与动态形状**: 同时支持固定 Batch Size 和可变 Batch Size 的场景。
- **可配置的 Tiling**: 针对 NPU 性能优化的可配置分块形状。
- **工程化实现**: 包含配置类（Dataclass）、完整的类型注解和详细的文档。

## 代码结构

- **`ffn_module.py`**: FFN 模块的核心实现与测试脚本，包含多个测试用例和示例用法。

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

### 执行测试

```bash
# 运行所有测试用例
python3 ffn_module.py

# 列出所有可用的测试用例
python3 ffn_module.py --list
```

## FFN 架构说明

FFN 模块实现了标准的 Transformer 前馈网络逻辑：

### 标准 FFN (GELU/ReLU)
```
Input [B, H]
  → Gate Projection [B, H] @ [H, I] → [B, I]
  → Activation (GELU/ReLU)
  → Down Projection [B, I] @ [I, H] → [B, H]
  → Output [B, H]
```

### SwiGLU FFN
```
Input [B, H]
  → Gate Projection [B, H] @ [H, I] → [B, I]
  → Up Projection [B, H] @ [H, I] → [B, I]
  → SwiGLU(Gate, Up) → [B, I]
  → Down Projection [B, I] @ [I, H] → [B, H]
  → Output [B, H]
```
其中：`B`=Batch Size, `H`=Hidden Size, `I`=Intermediate Size。

## 关键配置参数 (`FFNConfig`)

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `hidden_size` | `int` | 必填 | 隐藏层维度大小 |
| `intermediate_size` | `int` | 必填 | 中间层维度大小 |
| `activation` | `str` | `"gelu"` | 激活函数: `"gelu"`, `"swiglu"`, 或 `"relu"` |
| `dtype` | `pypto.DataType` | `DT_FP16` | 计算使用的数据类型 |
| `use_dynamic_shape` | `bool` | `False` | 是否支持动态 Batch Size |
| `vec_tile_shape` | `tuple` | `(64, 128)` | 向量运算的 Tiling 形状 |
| `cube_tile_shape` | `tuple` | `(64, 128, 128)` | 矩阵运算的 Tiling 形状 |
| `basic_batch` | `int` | `32` | 动态处理时的基础 Batch 大小 |

## 最佳实践
1. **性能调优**: 根据模型规模调整 `vec_tile_shape` 和 `cube_tile_shape`。较大的 Tile 通常能提供更好的计算密度，但会占用更多内部缓存。
2. **数据类型**: 对于 LLM 推理，推荐使用 `DT_BF16` 以获得更好的精度与性能平衡。
3. **动态 Batch**: 当推理服务的 Batch Size 频繁变化时，开启 `use_dynamic_shape` 并设置合理的 `basic_batch`。

## 注意事项
- 本模块目前主要支持 2D 张量输入。
- GELU 激活采用常用于高性能计算的近似实现。
- 请确保 NPU 上有足够的显存来容纳权重和中间张量。
