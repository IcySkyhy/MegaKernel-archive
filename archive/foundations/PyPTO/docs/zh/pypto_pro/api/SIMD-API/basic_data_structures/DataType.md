# pypto_pro.language.DataType

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

所有数据类型常量枚举。DataType自身还提供位宽、符号性等特征查询方法。

## 取值

| 常量 | 位宽 | 有符号 | 浮点 | C类型 | 类别 | 典型用途 |
|---|---|---|---|---|---|---|
| `pypto_pro.language.DT_BOOL` | 8 | 否 | 否 | `bool` | 布尔 | 掩码、条件判断 |
| `pypto_pro.language.DT_INT4` | 4 | 是 | 否 | — | 有符号整型 | 低精度量化 |
| `pypto_pro.language.DT_INT8` | 8 | 是 | 否 | `int8_t` | 有符号整型 | 量化输出（[-128, 127]） |
| `pypto_pro.language.DT_INT16` | 16 | 是 | 否 | `int16_t` | 有符号整型 | 索引、中间计算 |
| `pypto_pro.language.DT_INT32` | 32 | 是 | 否 | `int32_t` | 有符号整型 | 索引、累加器、标量参数 |
| `pypto_pro.language.DT_INT64` | 64 | 是 | 否 | `int64_t` | 有符号整型 | 大整数、地址计算、坐标/偏移 |
| `pypto_pro.language.DT_UINT4` | 4 | 否 | 否 | — | 无符号整型 | 低精度量化 |
| `pypto_pro.language.DT_UINT8` | 8 | 否 | 否 | `uint8_t` | 无符号整型 | 量化输出（[0, 255]）、bit-packed掩码 |
| `pypto_pro.language.DT_UINT16` | 16 | 否 | 否 | `uint16_t` | 无符号整型 | 索引 |
| `pypto_pro.language.DT_UINT32` | 32 | 否 | 否 | `uint32_t` | 无符号整型 | 字节偏移（`gatherb`） |
| `pypto_pro.language.DT_UINT64` | 64 | 否 | 否 | `uint64_t` | 无符号整型 | 大整数、地址计算 |
| `pypto_pro.language.DT_FP4` | 4 | 否 | 是 | — | IEEE浮点 | 低精度推理 |
| `pypto_pro.language.DT_FP8E4M3FN` | 8 | 否 | 是 | `float8_e4m3_t` | IEEE浮点 | FP8推理（E4M3格式） |
| `pypto_pro.language.DT_FP8E5M2` | 8 | 否 | 是 | `float8_e5m2_t` | IEEE浮点 | FP8推理（E5M2格式） |
| `pypto_pro.language.DT_FP8E8M0` | 8 | 否 | 是 | `float8_e8m0_t` | IEEE浮点 | MXFP8/MXFP4每组32个元素共享的scale |
| `pypto_pro.language.DT_FP4E2M1` | 4 | 否 | 是 | `float4_e2m1x2_t` | IEEE浮点 | E2M1格式的4位浮点数据，GM中两个元素打包为1字节 |
| `pypto_pro.language.DT_FP4E1M2` | 4 | 否 | 是 | `float4_e1m2x2_t` | IEEE浮点 | E1M2格式的4位浮点数据，GM中两个元素打包为1字节 |
| `pypto_pro.language.DT_FP16` | 16 | 否 | 是 | `half` | IEEE浮点 | 矩阵输入、向量计算（最常用） |
| `pypto_pro.language.DT_FP32` | 32 | 否 | 是 | `float` | IEEE浮点 | 累加器、高精度计算 |
| `pypto_pro.language.DT_BF16` | 16 | 否 | 是 | `bfloat16_t` | Brain浮点 | 矩阵输入（动态范围优于FP16） |
| `pypto_pro.language.DT_HF4` | 4 | 否 | 是 | — | 海思浮点 | 低精度推理 |
| `pypto_pro.language.DT_HF8` | 8 | 否 | 是 | `hifloat8_t` | 海思浮点 | 低精度推理 |

## 补充说明

**特征查询方法**：

| 方法 | 说明 |
|---|---|
| `dtype.get_bit()` / `dtype.bits()` | 取位宽（如`pypto_pro.language.DT_FP16.get_bit()`返回16） |
| `dtype.is_float()` | 是否浮点类型 |
| `dtype.is_int()` | 是否整型（有符号或无符号） |
| `dtype.is_signed_int()` / `dtype.is_signed()` | 是否有符号整型 |
| `dtype.is_unsigned_int()` / `dtype.is_unsigned()` | 是否无符号整型 |
| `dtype.to_string()` / `str(dtype)` | 人类可读名称（如`"fp16"`） |
| `dtype.to_c_type_string()` | C类型字符串（如`"half"`） |

**默认类型常量**：裸整数常量（如`42`）默认为`pypto_pro.language.DT_INT64`，裸浮点常量（如`3.14`）默认为`pypto_pro.language.DT_FP32`。

**MX浮点类型**：MXFP8的数据元素使用`DT_FP8E4M3FN`或`DT_FP8E5M2`，MXFP4的数据元素可使用`DT_FP4E2M1`或`DT_FP4E1M2`；两者均使用`DT_FP8E8M0`保存分组缩放因子。E8M0编码值`e`对应缩放因子`2^(e-127)`。类型组合和分组规则见[`matmul_mx`](../operation/matrix_computation/matmul_mx.md)和[`matmul_mx_acc`](../operation/matrix_computation/matmul_mx_acc.md)。

## 调用示例

```python
import pypto_pro.language as pl
# TileType 中指定数据类型
tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)

# Tensor 中指定数据类型
x: pl.Tensor[[64, 128], pl.DT_FP16]

# 特征查询
assert pl.DT_FP16.get_bit() == 16
assert pl.DT_FP16.is_float()
assert pl.DT_INT8.is_signed_int()
assert pl.DT_UINT8.is_unsigned_int()
```
