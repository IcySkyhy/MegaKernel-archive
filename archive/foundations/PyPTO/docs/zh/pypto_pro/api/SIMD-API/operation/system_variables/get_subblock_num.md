# pypto_pro.language.get_subblock_num

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

获取当前block的subblock总数（即一个block关联的从核数量，又称task ration）。

## 函数原型

```python
val = pypto_pro.language.get_subblock_num()
```

## 参数说明

无参数。

## 返回值说明

返回当前block的subblock总数，类型为整型Expr。返回值与核类型及编译模式有关：

- **AIC核**：始终返回1（AIC为block，无AIC从核）。
- **AIV核**：
  - 融合算子（mix，AIC:AIV = 1:2）：返回2（每个AICore含2个AIV从核）。
  - 纯Vector算子（aiv-only）：返回1（AIV为block，无subblock划分）。

## 调用示例

在融合算子中，`get_block_idx()`在AIV核上返回的是逻辑编号（`block_idx * subblock_num + subblock_idx`），通过除以`get_subblock_num()`可还原物理AICore编号，使cube与vector两侧用统一的`core_id`切分数据：

```python
import pypto_pro.language as pl

NUM_CORES = 2


@pl.jit(auto_mutex=True)
def matmul_example(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    num_cores = pl.get_block_num()
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    with pl.section_cube():
        for i in pl.range(core_id, a.shape[0] // 128, num_cores):
            ...
```

> [!NOTE]说明
> 该除法在AIC核上为`block_idx // 1`，在AIV核上为`(block_idx * 2 + subblock_idx) // 2`，两者均得到相同的AICore编号，因此cube与vector可共享同一`core_id`做数据切分。
