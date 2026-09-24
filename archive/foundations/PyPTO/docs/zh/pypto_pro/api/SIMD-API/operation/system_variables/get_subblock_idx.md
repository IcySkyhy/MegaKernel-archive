# pypto_pro.language.get_subblock_idx

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

获取当前逻辑AI Core内AIC或AIV的subblock索引。

## 函数原型

```python
val = pypto_pro.language.get_subblock_idx()
```

无参数。

## 返回值说明

返回设备运行时产生的整型标量值，可用于Kernel内整数运算和索引。取值范围为
`[0, get_subblock_num())`；在AIC与AIV比例为1:2的混合Kernel中，同一逻辑Block对应的两个AIV分别返回`0`和`1`。

## 典型使用场景

`pypto_pro.language.get_subblock_idx()`主要用于以下两种模式：

1. **`insert` + Cube模式**：每个子核计算部分结果，用`pypto_pro.language.insert`拼入Mat Tile（L1 NZ缓冲），Cube侧读取合并后的完整数据。详见[insert](../memory_data_movement/insert.md)文档示例。

2. **条件执行**：根据子核号决定是否执行某段代码，例如只让sub-core 0发起`pypto_pro.language.ssbuf_store`。

> [!CAUTION]注意
> 纯Vector Kernel中的两个子核共享MTE搬运管道，不能由每个子核分别使用`pypto_pro.language.store`向GM的不同区域写入数据。按子核切分数据搬运时，使用`insert` + Cube模式。

`get_block_idx()`用于Vector段的全局AIV数据分片，`get_subblock_idx()`用于区分同一逻辑Block内的不同AIV。

## 调用示例

下面示例读取subblock索引，并执行64×64 FP32逐元素加法。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def subblock_add_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    y: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_x = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_y = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_sum = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        _sub_index = pl.get_subblock_idx()
        cur_x = tile_x.current()
        cur_y = tile_y.current()
        cur_sum = tile_sum.current()
        pl.load(cur_x, x, [0, 0])
        pl.load(cur_y, y, [0, 0])
        pl.add(cur_sum, cur_x, cur_y)
        pl.store(out, cur_sum, [0, 0])
```
