# pypto_pro.language.system.set_cross_core / pypto_pro.language.system.wait_cross_core

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

跨核同步：跨AI Core的发信号 / 等信号。

## 函数原型

```python
pypto_pro.language.system.set_cross_core(*, pipe, event_id, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
pypto_pro.language.system.wait_cross_core(*, pipe, event_id, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `pipe` | 输入 | 发信号 / 等信号所在的pipe |
| `event_id` | 输入 | 事件id |
| `sync_mode` | 输入 | 同步模式，set / wait两侧须一致 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `pipe` | 输入 | 发信号 / 等信号所在硬件pipe，取值：<br>`pypto_pro.language.PipeType.MTE1` / `MTE2` / `MTE3` / `V` / `S` / `FIX`<br>set侧pipe为**发送核**执行set的pipe，wait侧pipe为**接收核**执行wait的pipe，两者分属不同核的不同pipe，无需相同。<br>典型组合：Cube→V通信用set(`FIX`) + wait(`V`)；V→Cube通信用set(`MTE3`/`V`) + wait(`MTE1`/`FIX`)。 |
| `event_id` | 输入 | 整型常量（静态）或运行时整型标量表达式（动态），有效取值为0～15。静态值由前端检查；动态表达式由调用方保证运行时不越界。配对的set和wait须使用相同事件号。 |
| `sync_mode` | 输入 | 同步模式，set侧和wait侧须使用相同的sync_mode，取值：<br>- `pl.CrossCoreSyncMode.INTER_BLOCK`（mode 0）：AI Core核间同步。AIC场景同步所有AIC核，AIV场景同步所有AIV核。<br>- `pl.CrossCoreSyncMode.INTER_SUBBLOCK`（mode 1）：AI Core内部AIV核间同步。<br>- `pl.CrossCoreSyncMode.INTRA_BLOCK`（mode 2，默认）：AI Core内部AIC与AIV之间同步。<br>- `pl.CrossCoreSyncMode.UNICAST_BLOCK`（mode 3）：AI Core内部AIC与AIV之间同步，AIV0与AIV1可单独触发AIC等待。 |

## 调用示例

`set_cross_core` / `wait_cross_core`用于Vector侧与Cube侧的跨核通信。下面是一个完整Kernel：Vector侧计算`x + y`并通过`insert`拼入L1 NZ缓冲，`set_cross_core`通知Cube侧；Cube侧`wait_cross_core`后读取并做matmul，计算`out = (x + y) @ rhs`。

```python
import pypto_pro.language as pl


@pl.jit()
def cross_core_kernel(
    x: pl.Tensor[[64, 64], pl.DT_FP32],
    y: pl.Tensor[[64, 64], pl.DT_FP32],
    rhs: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    v1_mat = pl.make_tile(
        pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat,
                    layout=pl.NZ),
        addr=0x10000, size=16384)

    with pl.section_vector():
        sub_index = pl.get_subblock_idx()
        off = sub_index * 32

        tile_x = pl.make_tile(pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                              addr=0x0000, size=8192)
        tile_y = pl.make_tile(pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                              addr=0x2000, size=8192)
        tile_sum = pl.make_tile(pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
                                addr=0x4000, size=8192)
        tile_nz = pl.make_tile(
            pl.TileType(shape=[32, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec,
                        layout=pl.NZ),
            addr=0x6000, size=8448)

        pl.load(tile_x, x, [off, 0])
        pl.load(tile_y, y, [off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.add(tile_sum, tile_x, tile_y)
        pl.move(tile_nz, tile_sum)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.insert(v1_mat, tile_nz, [off, 0])
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=2)

    with pl.section_cube():
        rhs_mat = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat,
                        layout=pl.NZ),
            addr=0x0000, size=16384)
        v1_left = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left,
                        layout=pl.NZ),
            addr=0x0000, size=16384)
        rhs_right = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right,
                        layout=pl.ZN),
            addr=0x0000, size=16384)
        c_l0c = pl.make_tile(
            pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                        layout=pl.NZ, fractal=1024),
            addr=0x0000, size=16384)

        pl.load(rhs_mat, rhs, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.move(rhs_right, rhs_mat)
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=2, sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK)
        pl.move(v1_left, v1_mat)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.matmul(c_l0c, v1_left, rhs_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.store(out, c_l0c, [0, 0])
```
