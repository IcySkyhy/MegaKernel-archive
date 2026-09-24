# pypto_pro.language.system.bar_m / pypto_pro.language.system.bar_mte1 / pypto_pro.language.system.bar_mte2 / pypto_pro.language.system.bar_mte3 / pypto_pro.language.system.bar_fix / pypto_pro.language.system.bar_all

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

Barrier同步：对M、MTE1、MTE2、MTE3、FIX中的指定pipe，或本AI Core的全部pipe做屏障，等待对应pipe上的前序操作完成。Barrier是单点同步操作。

## 函数原型

```python
pypto_pro.language.system.bar_m()
pypto_pro.language.system.bar_mte1()
pypto_pro.language.system.bar_mte2()
pypto_pro.language.system.bar_mte3()
pypto_pro.language.system.bar_fix()
pypto_pro.language.system.bar_all()
```

无参数。

## 流水类型

| API | 同步范围 | 调用位置 |
|---|---|---|
| `bar_m()` | 矩阵（M）流水线内barrier，等待前序所有矩阵操作完成 | Cube section |
| `bar_mte1()` | MTE1流水线内barrier，等待前序所有MTE1操作完成 | Cube section |
| `bar_mte2()` | MTE2流水线内barrier，等待前序所有MTE2操作完成 | Cube或Vector section |
| `bar_mte3()` | MTE3流水线内barrier，等待前序所有MTE3操作完成 | Cube或Vector section |
| `bar_fix()` | FIX流水线内barrier，等待前序所有FIX操作完成 | Cube section |
| `bar_all()` | 本AI Core内的全pipe barrier，等待V/M/MTE1/MTE2/MTE3/FIX等pipe的前序操作完成 | Cube或Vector section |

`bar_all()`仅同步当前AI Core内的全部pipe，不是多个AI Core之间的全局屏障；多核同步请使用[`sync_all`](sync_all.md)。

## 调用示例

### bar_all —— 循环体顶部全pipe barrier

在循环内load前插入`bar_all()`，确保上一轮store完成后再搬运新数据。

```python
import pypto_pro.language as pl


@pl.jit()
def bar_all_kernel(
    x: pl.Tensor[[128, 64], pl.DT_FP16],
    out: pl.Tensor[[128, 64], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_x = pl.make_tile(tt, addr=0x0000, size=8192)
    tile_out = pl.make_tile(tt, addr=0x2000, size=8192)
    with pl.section_vector():
        for i in pl.range(0, 128, 64):
            pl.system.bar_all()
            pl.load(tile_x, x, [i, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.add(tile_out, tile_x, tile_x)
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.store(out, tile_out, [i, 0])
```

### bar_m —— Cube（矩阵）流水线内barrier

在`section_cube()`中，两次matmul之间用`bar_m()`同步，确保前一次矩阵乘完成后再开始下一次。下面是一个完整Kernel：两次matmul之间插入`bar_m()`，第二次覆盖acc，最终`out = a @ b`。

```python
import pypto_pro.language as pl

TILE = 64


@pl.jit(auto_mutex=True)
def bar_m_kernel(
    a: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP16],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x2000, mutex_ids=[1])
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[2])
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[3])
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[4])

    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.system.bar_m()
        pl.matmul(ac, al, br)
        pl.store(out, ac, [0, 0])
```

`bar_m`只约束Matmul流水内部的先后关系；跨流水依赖仍应使用对应的同步接口或`auto_mutex`。

该接口不替代跨核同步。
