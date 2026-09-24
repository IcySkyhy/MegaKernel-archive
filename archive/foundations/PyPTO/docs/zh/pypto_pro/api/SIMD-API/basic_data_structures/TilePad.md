# pypto_pro.language.TilePad

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->

## 功能说明

Tile边界不足时的填充方式枚举，用于尾块/非满块场景。

当tile的有效数据区域小于其shape时（如动态维度的尾块），超出有效区域的部分需要按指定模式填充。

## 取值

| 取值 | 说明 | 典型用途 |
|---|---|---|
| `pypto_pro.language.TilePad.null` | 不填充（默认） | 大多数场景 |
| `pypto_pro.language.TilePad.zero` | 补0 | 卷积padding、零初始化 |
| `pypto_pro.language.TilePad.max` | 补该类型最大值 | 取最小值操作的无效区域 |
| `pypto_pro.language.TilePad.min` | 补该类型最小值 | flash attention掩码（无效行补FP32 min，被max/softmax忽略） |

## 补充说明

**flash attention掩码**：当KV长度不是tile大小的整数倍时，最后一块的无效行需要补FP32最小值，这样在后续的`row_max`和`exp`操作中会被自然忽略。

**卷积padding**：卷积操作中边界填充零值。

## 调用示例

```python
import pypto_pro.language as pl
# 不填充（默认）
tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16,
                 target_memory=pl.MemorySpace.Vec)

# 补零（卷积 padding）
tt_pad = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16,
                     target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.zero)

# 补最小值（flash attention 掩码）
tt_mask = pl.TileType(shape=[64, 128], dtype=pl.DT_FP32,
                      target_memory=pl.MemorySpace.Vec, pad=pl.TilePad.min)
```
