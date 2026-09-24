# pypto_pro.language.system.set_mm_layout_transform

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

切换matmul的fixpipe结果读出方向，开启后fixpipe沿N方向（列优先）从L0C读数据。

## 函数原型

```python
pypto_pro.language.system.set_mm_layout_transform(*, enabled)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `enabled` | 输入 | 是否开启布局变换 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `enabled` | 输入 | `True`：开启N方向读出（fixpipe沿列优先从L0C读数据）<br>`False`：恢复M方向读出（默认，fixpipe沿行优先）<br>仅在matmul K维分块累加场景中使用，须成对出现：K循环前`enabled=True`，写回GM后`enabled=False`<br>非K累加场景（单次matmul）不需要此开关 |

## 调用示例

`set_mm_layout_transform`仅在matmul K维分块累加场景中使用。下面是一个完整kernel：计算`C[128,128] = A[128,256] @ B[256,128]`，K维按128分块累加。K循环前`enabled=True`，写回GM后`enabled=False`。

```python
import pypto_pro.language as pl

TILE_ACC = 128
K_SIZE_ACC = 256


@pl.jit(auto_mutex=True)
def mm_layout_kernel(
    a: pl.Tensor[[TILE_ACC, K_SIZE_ACC], pl.DT_FP16],
    b: pl.Tensor[[K_SIZE_ACC, TILE_ACC], pl.DT_FP16],
    c: pl.Tensor[[TILE_ACC, TILE_ACC], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_ACC, TILE_ACC], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_ACC, TILE_ACC], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_ACC, TILE_ACC], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left,
                         layout=pl.NZ),
        addrs=0x0000, mutex_ids=[4, 5])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_ACC, TILE_ACC], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[6, 7])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_ACC, TILE_ACC], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         fractal=1024),
        addrs=0x0000, mutex_ids=[8])

    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        ac = acc.current()
        for k in pl.range(0, K_SIZE_ACC, TILE_ACC):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_left.next()
            br = b_right.next()
            pl.load(cur_a, a, [0, k])
            pl.load(cur_b, b, [k, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if k == 0:
                pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(c, ac, [0, 0], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)
```
