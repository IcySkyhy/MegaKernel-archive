# pypto_pro.language.trap

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

插入trap指令强制中止执行，用于调试。无条件中止，不接受任何参数。

## 函数原型

```python
pypto_pro.language.trap()
```

无参数。

## 流水类型

S（标量流水）。

## 调用示例

```python
import pypto_pro.language as pl
# 条件中止
if flag:
    pl.trap()

# 无条件中止（调试用）
pl.trap()
```
