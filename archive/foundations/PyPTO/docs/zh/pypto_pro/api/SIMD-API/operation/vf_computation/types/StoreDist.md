# StoreDist

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

StoreDist定义了[`vf.store_align`](../data_movement/store_align.md)的数据存储分布模式，用于控制从寄存器到UB的数据搬运方式。

## 原型定义

```python
class StoreDist(enum.Enum):
     # reg_tensor单搬出模式
     NORM = ...  # 普通对齐存储（默认），64位宽数据类型DT_INT64/DT_UINT64只支持此模式
     NORM_B16 = ...  # 按B16粒度普通存储
     FIRST_ELEMENT = ...  # 仅存储lane 0（首个元素）
     PACK = ...  # 压缩存储，根据mask将src中有效元素的低半部分bit数据连续存储于dst中
     PACK4 = ...  # 4元素压缩存储，将有效元素的低8bit数据连续存储
     # reg_tensor双搬出模式
     INTLV = ...  # 交错存储，将src0、src1中的元素交错存储（根据dtype自动选择B8/B16/B32）
     INTLV_B32 = ...  # 按B32粒度交错存储
     # mask_tensor模式
     # NORM 同上，搬运VL/8数据
     # PACK 同上，每间隔1bit舍弃数据，将VL/8的数据压缩为VL/16搬出
```

## 约束说明

StoreDist的取值根据目标模式不同，支持的模式有所区别：

- **reg_tensor单搬出模式**：支持 `NORM`、`NORM_B16`、`FIRST_ELEMENT`、`PACK`、`PACK4`
- **reg_tensor双搬出模式**：支持 `INTLV`、`INTLV_B32`（需要两个源寄存器）
- **mask_tensor模式**：支持 `NORM`、`PACK`

## 调用示例

```python
import pypto_pro.language as pl

@pl.vector_function
def vf_kernel():
    vf.store_align(ub_tile, reg, dist=pl.StoreDist.NORM)
```
