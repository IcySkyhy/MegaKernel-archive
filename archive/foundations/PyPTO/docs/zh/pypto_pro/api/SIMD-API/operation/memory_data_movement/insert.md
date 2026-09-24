# pypto_pro.language.insert

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

把一块较小的源Tile，按`offset=[row, col]`指定的行列位置，嵌入到一块较大的目标Tile中。对应pto-isa的TINSERT指令，用于**UB→L1**的搬运，典型场景是把UB上的向量计算结果按NZ格式拼入L1缓冲区，供后续Cube计算使用。

源Tile的左上角对齐到目标Tile的`offset`位置。`offset[0]`为目标Tile的行偏移`row`，`offset[1]`为列偏移`col`。

## 函数原型

```python
pypto_pro.language.insert(dst_tile, src_tile, offset)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_tile` | 输出 | 目标Tile，`target_memory`为`pl.MemorySpace.Mat`（L1），通常采用NZ格式 |
| `src_tile` | 输入 | 源Tile，`target_memory`为`pl.MemorySpace.Vec`（UB） |
| `offset` | 输入 | 长度为2的序列，元素为整型常量或运行时整型标量表达式 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_tile` | 输出 | 数据类型：b8、b16、b32、b64<br>`target_memory`须为`pl.MemorySpace.Mat`（L1）；首地址必须32字节对齐 |
| `src_tile` | 输入 | 数据类型：b8、b16、b32、b64<br>`target_memory`须为`pl.MemorySpace.Vec`（UB） |
| `offset` | 输入 | 格式为`[row, col]`；须满足`row + src行数 ≤ dst行数`，`col + src列数 ≤ dst列数`，否则越界 |

## 流水类型

MTE3（UB → L1的搬运流水）。

## 调用示例

下面是一个完整Kernel：Vector侧每个subcore处理32行，算出`x+y`后用`pypto_pro.language.move`做ND→NZ转换，再用`pypto_pro.language.insert`按二维offset拼入一块64×64的L1 NZ缓冲`v1_mat`；Cube侧把`v1_mat`当左矩阵和`rhs`做matmul。`insert`在此承担UB→L1的NZ拼接。

注意：`insert`的源Tile必须是NZ格式（`layout=pl.NZ`），需先经`pypto_pro.language.move`把ND结果转成NZ。示例使用`make_tile_group`管理Tile资源，并通过`auto_mutex`完成组内流水同步；Vector与Cube之间仍显式使用`set_cross_core`/`wait_cross_core`，以`INTRA_BLOCK`模式完成AIV→AIC的段间同步。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def insert_matmul_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    y: pl.Tensor[[64, 64], pl.DT_FP32],
    rhs: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    v1_mat_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat,
                         layout=pl.NZ),
        addrs=0x10000, mutex_ids=[0])

    with pl.section_vector():
        sub_index = pl.get_subblock_idx()
        off = sub_index * 32

        tile_x_group = pl.make_tile_group(
            type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x0000, mutex_ids=[1])
        tile_y_group = pl.make_tile_group(
            type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x2000, mutex_ids=[2])
        tile_sum_group = pl.make_tile_group(
            type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x4000, mutex_ids=[3])
        tile_nz_group = pl.make_tile_group(
            type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                             layout=pl.NZ),
            addrs=0x6000, mutex_ids=[4])
        v1_mat = v1_mat_group.current()
        tile_x = tile_x_group.current()
        tile_y = tile_y_group.current()
        tile_sum = tile_sum_group.current()
        tile_nz = tile_nz_group.current()

        pl.load(tile_x, x, [off, 0])
        pl.load(tile_y, y, [off, 0])

        pl.add(tile_sum, tile_x, tile_y)
        pl.move(tile_nz, tile_sum)   # ND -> NZ

        pl.insert(v1_mat, tile_nz, [off, 0])   # UB -> L1 NZ2NZ
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=2)

    with pl.section_cube():
        rhs_mat_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat,
                             layout=pl.NZ),
            addrs=0x0000, mutex_ids=[5])
        v1_left_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left,
                             layout=pl.NZ),
            addrs=0x0000, mutex_ids=[6])
        rhs_right_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right,
                             layout=pl.ZN),
            addrs=0x0000, mutex_ids=[7])
        c_l0c_group = pl.make_tile_group(
            type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                             layout=pl.NZ, fractal=1024),
            addrs=0x0000, mutex_ids=[8])
        v1_mat = v1_mat_group.current()
        rhs_mat = rhs_mat_group.current()
        v1_left = v1_left_group.current()
        rhs_right = rhs_right_group.current()
        c_l0c = c_l0c_group.current()

        pl.load(rhs_mat, rhs, [0, 0])
        pl.move(rhs_right, rhs_mat)

        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.move(v1_left, v1_mat)

        pl.matmul(c_l0c, v1_left, rhs_right)

        pl.store(out, c_l0c, [0, 0])
```

其他典型用法（节选）：

```python
# 两个维度均有偏移
pl.insert(p_mat_slot, p_f16_back_slot, [TKV // 2, TS_HALF * sub_id])

# 仅沿第 0 维偏移
pl.insert(v1_mat, tile_nz, [off, 0])
```
