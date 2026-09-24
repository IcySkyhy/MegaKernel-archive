# pypto_pro.language.dump_data

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

调测打印接口，用于打印GM Tensor或Tile的内容，支持全量打印和窗口打印。

- 输入为`Tensor`（GM全局内存张量）时，打印GM上的Tensor数据
- 输入为`Tile`（通过`make_tile`/`make_tile_group`分配）时，打印Tile数据

Tile通过[`pypto_pro.language.make_tile`](../../SIMD-API/operation/resource_management/make_tile.md)或[`pypto_pro.language.make_tile_group`](../../SIMD-API/operation/resource_management/make_tile_group.md)创建。不同内存空间的Tile打印能力不同：

- `Vec`(UB) Tile：由后端直接通过`TPRINT`打印，无需`workspace`
- `Acc`(L0C) Tile：须通过`workspace`参数中转打印（先将Tile数据写回GM，再打印）
- `Mat`(L1)/`Left`(L0A)/`Right`(L0B) Tile：无法直接打印

打印结果直接输出到终端。

## 函数原型

```python
pypto_pro.language.dump_data(data, offsets=None, shapes=None, *, workspace=None, loc=False)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `data` | 输入 | 要打印的数据，可以是Tensor或Tile |
| `offsets` | 输入 | 可选，窗口起始偏移（各维） |
| `shapes` | 输入 | 可选，窗口大小（各维） |
| `workspace` | 输入 | 可选，GM上的临时Tensor，仅用于Acc Tile的中转打印 |
| `loc` | 输入 | 可选，是否在输出前打印源文件/行号 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `data` | 输入 | 必须是Tensor（TensorType）或Tile（TileType），其他类型报`TypeError` |
| `offsets` | 输入 | 由整型常量或运行时整型标量表达式组成的序列，长度须等于数据的维数；须与`shapes`同时提供或同时为`None`。Tile窗口当前仅支持二维Tile |
| `shapes` | 输入 | 由整型常量或运行时整型标量表达式组成的序列，长度须等于数据的维数；须与`offsets`同时提供或同时为`None`。其中编译期常量必须大于0；Tensor窗口模式还要求最内维stride为编译期常量1。Tile窗口要求二维且Tile物理shape为编译期常量 |
| `workspace` | 输入 | 仅当`data`为Acc Tile时有效；Tensor传入该参数会报`ValueError`，其他内存空间的Tile会在IR校验时报错。必须是与Tile dtype相同的GM `TensorType`。后端会先将完整Acc Tile写入workspace，即使只打印窗口，workspace也必须至少容纳完整物理Tile；当前仅支持二维、静态物理shape的Acc Tile |
| `loc` | 输入 | `True`或`False`（默认） |

当`offsets`和`shapes`均为`None`（默认）时，打印整个数据的全部数据。

### Acc（L0C）Tile dump

Acc（L0C）Tile无法像Vec Tile那样直接通过`TPRINT`打印。`dump_data`通过`workspace`参数提供一条中转路径：先将Tile数据写回GM上的`workspace` Tensor，再由`TPRINT`打印该Tensor。

使用约束：

- `workspace`必须是`TensorType`，可以是核函数参数中的`pl.Tensor`，也可以通过`pl.make_tensor`从`pl.Ptr`构造，不能是Tile
- `workspace`的dtype必须与待dump的Tile dtype一致
- 无论全量dump还是窗口dump，后端都会先将完整Acc Tile写入`workspace`，因此其容量须至少覆盖完整物理Tile
- Acc Tile必须为二维，且物理shape必须是编译期常量

## 流水类型

V（向量流水）。

## 调用示例

### Tensor输入

```python
import pypto_pro.language as pl


@pl.jit()
def dump_data_tensor_full_kernel(
    out: pl.Tensor[[16], pl.DT_INT32],
):
    with pl.section_vector():
        for i in pl.range(0, 16):
            out[i] = i * 10
        pl.dump_data(out)
```

全量dump输出示例：

```text
=== [TPRINT GlobalTensor] Data Type: int32, Layout: ND ===
  Shape: [1, 1, 1, 16]
  Batch [0, 0, 0]:
      0       10       20       30       40       50       60       70
     80       90      100      110      120      130      140      150
```

动态偏移示例：

```python
# 循环变量作为偏移（动态）
for i in pl.range(0, 16, 4):
    pl.dump_data(out, offsets=[i], shapes=[4])

# get_block_idx() 作为偏移（动态）
vidx = pl.get_block_idx()
pl.dump_data(out, offsets=[vidx * 4], shapes=[4])
```

### Tile Vec输入

`Vec`(UB) Tile可直接打印，无需`workspace`。`Mat`(L1)/`Left`(L0A)/`Right`(L0B) Tile无法直接打印。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def dump_data_tile_full_kernel(
    a: pl.Tensor[[32, 32], pl.DT_INT32],
    b: pl.Tensor[[32, 32], pl.DT_INT32],
    out: pl.Tensor[[32, 32], pl.DT_INT32],
):
    tt = pl.TileType(shape=[32, 32], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    ta_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tb_group = pl.make_tile_group(type=tt, addrs=0x1000, mutex_ids=[1])
    tc_group = pl.make_tile_group(type=tt, addrs=0x2000, mutex_ids=[2])
    with pl.section_vector():
        ta = ta_group.current()
        tb = tb_group.current()
        tc = tc_group.current()
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        pl.add(tc, ta, tb)
        pl.dump_data(tc)
        pl.store(out, tc, [0, 0])
```

全量dump输出示例：

```text
=== [TPRINT Tile] Data Type: int32, Layout: ND, TileType: Vec ===
  Source Shape: [32, 32]
      0       2       4       6       8      10      12      14
     64      66      68      70      72      74      76      78
    128     130     132     134     136     138     140     142
    192     194     196     198     200     202     204     206
    256     258     260     262     264     266     268     270
    320     322     324     326     328     330     332     334
    384     386     388     390     392     394     396     398
    448     450     452     454     456     458     460     462
```

### Acc（L0C）Tile输入（需要workspace）

Acc（L0C）Tile需要通过`workspace`参数提供GM上的临时Tensor进行中转。

全量dump：

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def dump_data_tile_acc_fp16_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16],
    b: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP32],
    workspace: pl.Tensor[[64, 64], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x0000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x2000, mutex_ids=[1])
    a_l0a = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[2])
    b_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
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
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.move(al, cur_a)
        pl.move(br, cur_b)
        pl.matmul(ac, al, br)

        # 打印整个 Acc Tile（需要 workspace 中转）
        pl.dump_data(ac, workspace=workspace)

        # 打印窗口（带 offsets/shapes 和 workspace）
        pl.dump_data(ac, offsets=[16, 16], shapes=[8, 8], workspace=workspace)

        pl.store(out, ac, [0, 0])
```

窗口dump（带`offsets`/`shapes`和`workspace`）：

```python
# workspace 声明为 pl.Tensor[[32, 32], pl.DT_FP32]
# 打印 Acc Tile 中偏移 [16, 16]、大小 8x8 的窗口
pl.dump_data(ac, offsets=[16, 16], shapes=[8, 8], workspace=workspace)
```

窗口模式输出示例（`offsets=[16, 16], shapes=[8, 8]`）：

```text
=== [TPRINT Acc Tile Window] Data Type: float32, Layout: NZ, TileType: Acc ===
  Source Shape: [64, 64], Window Offsets: [16, 16], Requested Shape: [8, 8], Valid Shape: [8, 8]
  3.2298 -15.6709  -9.2879 -12.3104 -16.4727  14.4406   3.6175 -14.7739
 -5.8024  -9.9790   8.3985   2.1331   0.1348 -12.8909   0.1361  -3.0442
  2.4974   1.5251 -13.1748   5.9634   6.1657   0.8389  13.8052 -11.0019
 -7.0973   2.4425  -3.5134   0.2277 -12.4192 -14.7564   6.8491 -15.4198
  4.8601  -0.9246  -5.6728  -4.2165  12.5482  -3.1285   8.9953  -8.2000
 15.5239  -6.9626  -2.0870   5.0846   0.7985  -0.8446  -4.7134  15.3558
  9.5928   9.8900  -1.1198   7.5672   0.0275   8.6235   0.7186   6.4263
 -2.8095  -1.8578   1.4832  -7.0184   8.0429   4.5278   1.3108  16.0369
```

在循环中使用动态偏移：

```python
for i in pl.range(0, 256, 64):
    for j in pl.range(0, 256, 64):
        # ... matmul ...
        pl.dump_data(ac, offsets=[8, 8], shapes=[32, 32], workspace=workspace)
        pl.store(out, ac, [i, j])
```
