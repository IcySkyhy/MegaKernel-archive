# pypto_pro.language.section_vector / section_cube

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

`section_vector`创建Vector区域，`section_cube`创建Cube区域。区域中的操作分别下沉到向量核或矩阵核执行。

- **`section_vector`**：用于UB（`MemorySpace.Vec`）tile上的向量计算，包括GM↔UB搬运（`load`/`store`）、逐元素运算、归约、转置等。
- **`section_cube`**：用于涉及L1/L0A/L0B/L0C以及E8M0分组缩放因子Tile的矩阵计算，包括GM→L1搬运（`load`）、L1→L0A/L0B/ScaleLeft/ScaleRight搬运（`move`）、矩阵乘（`matmul`/`matmul_mx`）、L0C→GM回写（`store`）。

同一kernel中可先后使用多个`section_vector` / `section_cube`，例如先在Cube区域完成matmul，再在Vector区域对结果做逐元素激活。

## 函数原型

```python
pypto_pro.language.section_vector()
pypto_pro.language.section_cube()
```

两者均为context manager，通过`with`语句使用，不接受参数。

## 调用示例

### section_cube矩阵乘

下面是一个完整kernel：在`section_cube`区域中从GM载入FP16输入到L1，move到L0A/L0B，用`matmul`完成矩阵乘后从L0C写回GM。本例启用`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def with_section_cube_fp16_kernel(
    x: pl.Tensor[[64, 32], pl.DT_FP16],
    y: pl.Tensor[[32, 64], pl.DT_FP16],
    z: pl.Tensor[[64, 64], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[1])
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 32], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[2])
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[32, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[3])
    c_l0c = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[4])
    with pl.section_cube():
        cur_a = a_l1.current()
        cur_b = b_l1.current()
        al = a_l0a.current()
        br = b_l0b.current()
        ac = c_l0c.current()
        pl.load(cur_a, x, [0, 0])
        pl.load(cur_b, y, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)
        pl.store(z, ac, [0, 0])
```

### section_vector逐元素计算

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def with_section_vector_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP32],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.relu(cur_out, cur_a)
        pl.store(out, cur_out, [0, 0])
```

### Cube与Vector串联

同一kernel中可先Cube后Vector，常用于matmul后接逐元素计算。Cube结果先写回GM，再由Vector区域读取时，须使用跨核同步保证先写后读：

```python
with pl.section_cube():
    pl.matmul(ac, al, br)
    pl.store(ws, ac, [0, 0])
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

with pl.section_vector():
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)
    pl.load(tile, ws, [0, 0])
    pl.relu(tile, tile)
    pl.store(out, tile, [0, 0])
```
