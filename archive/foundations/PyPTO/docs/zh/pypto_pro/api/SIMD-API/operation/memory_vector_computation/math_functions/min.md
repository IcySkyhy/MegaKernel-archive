# pypto_pro.language.min

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

取两个标量中的较小值。Python内置的`min()`在前端解析时会被自动转换为`pypto_pro.language.min`，两者等价。

> [!CAUTION]注意
> Tile逐元素取最小值使用[`pypto_pro.language.minimum`](../elementwise/minimum.md)。

## 函数原型

```python
result = pypto_pro.language.min(lhs, rhs)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `lhs` | 输入 | 左操作数（标量或数值常量） |
| `rhs` | 输入 | 右操作数（标量或数值常量） |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `lhs` | 输入 | Python `int`、Python `float`或Kernel内整型或浮点型标量表达式 |
| `rhs` | 输入 | Python `int`、Python `float`或Kernel内整型或浮点型标量表达式<br>两个操作数须属于同一数值类别（同为整型或同为浮点型），同类别内自动提升宽度（如INT32 + INT64 → INT64） |

## 流水类型

S（标量流水）

## 调用示例

```python
import pypto_pro.language as pl

# 循环边界计算（尾块处理）
m_size = pl.min(m_dim - i, 64)
n_size = pl.min(n_dim - j, 128)

# 与整数常量比较
causal_kv_tiles = pl.min(qi + 1, skv_tiles)

# 标量变量之间
c = pl.min(a, b)

# 浮点标量（两个操作数须同为浮点型）
scale = pl.min(pl.const(2.5, pl.DT_FP32), pl.const(-1.5, pl.DT_FP32))
```
