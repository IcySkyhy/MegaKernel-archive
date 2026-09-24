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

在Kernel代码中可直接使用Python内置`min(lhs, rhs)`，前端会自动将其解析为`pypto_pro.language.min(lhs, rhs)`，取两个标量中的较小值。

## 函数原型

```python
# 以下两种写法等价
result = min(lhs, rhs)
result = pypto_pro.language.min(lhs, rhs)
```

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `lhs` | 输入 | 左操作数（Python `int`、Python `float`或Kernel内整型或浮点型标量表达式） |
| `rhs` | 输入 | 右操作数（Python `int`、Python `float`或Kernel内整型或浮点型标量表达式） |

## 调用示例

```python
import pypto_pro.language as pl

@pl.jit()
def example_kernel(...):
    m_size = min(m_dim - i, 64)
    n_size = pl.min(n_dim - j, 128)

    causal_kv_tiles = min(qi + 1, skv_tiles)
    bottom = min(a, 0)
```

## 注意事项

- **仅用于标量**：用于循环边界、索引计算等场景
- **Tile逐元素取最小值**：使用[`pl.minimum`](../../SIMD-API/operation/memory_vector_computation/elementwise/minimum.md)
- **不支持多参数**：仅接受恰好2个参数，`min(a, b, c)`不支持
- **同类别约束**：两个操作数须同为整型或同为浮点型，混合int/float会报错

详见[`pl.min`接口文档](../../SIMD-API/operation/memory_vector_computation/math_functions/min.md)。
