# pypto_pro.language.get_block_idx

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

获取当前执行域中逻辑AI Core的全局索引，用于多核控制和数据偏移计算。

## 函数原型

```python
val = pypto_pro.language.get_block_idx()
```

无参数。

## 返回值说明

返回设备运行时产生的整型标量值，可用于Kernel内整数运算和索引。取值范围与Kernel的执行域有关：

- 仅启动Cube（AIC）或仅启动Vector（AIV）时，范围为`[0, get_block_num())`。
- 同时启动AIC与AIV时，AIC侧范围为`[0, get_block_num())`；AIV侧范围为`[0, get_subblock_num() * get_block_num())`。当前1:2配置下，AIV侧范围为`[0, 2 * get_block_num())`。

在Vector段中，该接口返回展平后的全局AIV逻辑索引，可直接用于数据分片和偏移计算。混合Kernel在Vector段做跨步切分时，工作单元总数为`get_block_num() * get_subblock_num()`。

## 调用示例

下面是一个仅包含Vector段的多核Kernel：用`kernel[None, NUM_CORES](...)`启动2个逻辑Block，每个AIV用`pypto_pro.language.get_block_idx()`获取全局逻辑索引，并处理64行逐元素加法。

```python
import pypto_pro.language as pl

NUM_CORES = 2


@pl.jit(auto_mutex=True)
def multicore_add_kernel(
    x: pl.Tensor[[128, 128], pl.DT_FP16],
    y: pl.Tensor[[128, 128], pl.DT_FP16],
    z: pl.Tensor[[128, 128], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_c = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        vidx = pl.get_block_idx()              # 当前AIV的全局逻辑索引
        _bnum = pl.get_block_num()             # 启动时配置的逻辑Block数
        offset = vidx * 64                     # 第vidx个AIV处理[vidx*64, +64)行
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_c = tile_c.current()
        pl.load(cur_a, x, [offset, 0])
        pl.load(cur_b, y, [offset, 0])
        pl.add(cur_c, cur_a, cur_b)
        pl.store(z, cur_c, [offset, 0])
```
