# ReduceMode

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持
<!-- end id3 -->

## 功能说明

ReduceMode定义了归约操作的执行模式，用于指定多线程或多设备环境下的归约计算方式，确保计算结果的正确性和性能。

## 原型定义

```python
class ReduceMode(enum.Enum):
     ATOMIC_ADD = ...  # 原子加法归约，使用原子操作确保多线程安全的数据累加
```
