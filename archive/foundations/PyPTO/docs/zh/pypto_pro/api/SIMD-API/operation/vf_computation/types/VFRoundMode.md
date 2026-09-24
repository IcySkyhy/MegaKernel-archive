# VFRoundMode

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

VFRoundMode定义了[`vf.astype`](../type_conversion/astype.md)类型转换时的浮点舍入模式。仅在可能导致精度损失且支持该舍入模式的转换中生效，不涉及精度损失的转换路径标记为`UNKNOWN`（可省略）。

不同转换路径支持的舍入模式不同，详见[`vf.astype`约束说明](../type_conversion/astype.md#约束说明)各表。

## 原型定义

```python
class VFRoundMode(enum.Enum):
     CAST_ROUND = ...  # 四舍五入，向远离零的方向舍入
     CAST_RINT = ...  # 默认舍入，向最近的偶数舍入
     CAST_FLOOR = ...  # 向下取整（floor）
     CAST_CEIL = ...  # 向上取整（ceil）
     CAST_TRUNC = ...  # 向零舍入（截断）
     CAST_ODD = ...  # 冯·诺伊曼舍入（最近的奇数）
     CAST_HYBRID = ...  # 混合舍入（仅Ascend 950PR/DT支持）
```

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    dst = vf.astype(src, dtype=pl.DT_INT32, round_mode=pl.VFRoundMode.CAST_FLOOR)
```
