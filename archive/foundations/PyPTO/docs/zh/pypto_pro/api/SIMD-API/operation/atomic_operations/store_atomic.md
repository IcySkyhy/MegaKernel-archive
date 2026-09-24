# pypto_pro.language.store（atomic）

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

原子累加写回是[`pypto_pro.language.store`](../memory_data_movement/store.md)的`atomic`参数用法之一，无需单独API。多核累加到同一GM区域时，通过`atomic=pl.AtomicType.AtomicAdd`启用硬件原子写，避免数据竞争。

完整的参数说明、约束与调用示例见[`pypto_pro.language.store`](../memory_data_movement/store.md)（重点关注`atomic`参数）。
