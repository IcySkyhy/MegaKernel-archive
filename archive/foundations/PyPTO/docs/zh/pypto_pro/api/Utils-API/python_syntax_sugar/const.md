# pypto_pro.language.const

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

创建指定数据类型的编译期常量标量。用于需要显式指定类型的标量值场景，如与`pypto_pro.language.DT_INT32`类型的Tile做比较、初始化索引值等。

## 函数原型

```python
result = pypto_pro.language.const(value, dtype)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `value` | 输入 | 数值常量（int或float） |
| `dtype` | 输入 | 目标数据类型 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `value` | 输入 | Python `int`或`float`常量值 |
| `dtype` | 输入 | `pypto_pro.language.DT_INT8`、`pypto_pro.language.DT_INT16`、`pypto_pro.language.DT_INT32`、`pypto_pro.language.DT_INT64`、`pypto_pro.language.DT_UINT8`、`pypto_pro.language.DT_UINT16`、`pypto_pro.language.DT_UINT32`、`pypto_pro.language.DT_UINT64`、`pypto_pro.language.DT_FP16`、`pypto_pro.language.DT_FP32`、`pypto_pro.language.DT_BF16`、`pypto_pro.language.DT_BOOL`等 |

## 返回值

标量值（`Scalar`），类型由`dtype`指定。

## 调用示例

```python
import pypto_pro.language as pl
# 创建 INT64 类型的零常量
zero_idx = pl.const(0, pl.DT_INT64)

# 创建 INT32 类型的常量
offset = pl.const(42, pl.DT_INT32)

# 创建 FP16 类型的常量
scale = pl.const(1.0, pl.DT_FP16)

# 创建负值常量
neg_offset = pl.const(-1, pl.DT_INT32)
```
