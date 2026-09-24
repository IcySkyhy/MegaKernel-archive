# pypto_pro.language.matmul_mx

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

完成一次带E8M0分组缩放因子的MX矩阵乘法：

```text
dst = dequant(lhs, scale_a) @ dequant(rhs, scale_b)
```

MX（Microscaling）矩阵计算采用分组缩放：矩阵数据沿K方向每连续32个元素为一组，每组共享一个E8M0缩放因子，该缩放因子称为MX scale，取值为`2^(e8m0-127)`。

MXFP8尾数支持E4M3与E5M2，MXFP4尾数支持E2M1与E1M2。同一侧的两种格式可以交叉组合；MXFP8与MXFP4不能混合使用。

数据通路由Left（L0A）及其配套的ScaleLeft缓冲区、Right（L0B）及其配套的ScaleRight缓冲区共同参与，计算结果写入Acc（L0C）。K维分块的首块使用本接口，后续块使用[`matmul_mx_acc`](matmul_mx_acc.md)。

## 函数原型

```python
pypto_pro.language.matmul_mx(
    dst_tile, lhs_tile, rhs_tile, scale_a, scale_b, *, phase=None
)
```

## 参数类型


| 参数       | 输入/输出 | 说明                                                          |
| ---------- | --------- | ------------------------------------------------------------- |
| `dst_tile` | 输出      | Acc/L0C Tile，保存FP32计算结果                                |
| `lhs_tile` | 输入      | Left/L0A Tile，MXFP8或MXFP4左矩阵尾数                         |
| `rhs_tile` | 输入      | Right/L0B Tile，MXFP8或MXFP4右矩阵尾数                        |
| `scale_a`  | 输入      | ScaleLeft Tile，A矩阵的E8M0 scale                             |
| `scale_b`  | 输入      | ScaleRight Tile，B矩阵的E8M0 scale                            |
| `phase`    | 输入      | 可选，K维分块时使用`pl.AccPhase.Partial`或`pl.AccPhase.Final` |

## 参数范围


| 参数       | 约束                                                                                                                 |
| ---------- | -------------------------------------------------------------------------------------------------------------------- |
| `dst_tile` | dtype必须为`DT_FP32`；shape为`[M,N]`；内存空间必须为`MemorySpace.Acc`                                                |
| `lhs_tile` | dtype为`DT_FP8E4M3FN`、`DT_FP8E5M2`、`DT_FP4E2M1`或`DT_FP4E1M2`；shape为`[M,K]`；内存空间必须为`MemorySpace.Left`  |
| `rhs_tile` | 与`lhs_tile`同属MXFP8组或MXFP4组；shape为`[K,N]`；内存空间必须为`MemorySpace.Right`                                  |
| `scale_a`  | dtype必须为`DT_FP8E8M0`；shape为`[M,K/32]`；内存空间必须为`MemorySpace.ScaleLeft`，默认`layout=pl.ZZ`、`fractal=32`  |
| `scale_b`  | dtype必须为`DT_FP8E8M0`；shape为`[K/32,N]`；内存空间必须为`MemorySpace.ScaleRight`，默认`layout=pl.NN`、`fractal=32` |

补充约束：

- `lhs_tile`的第二维与`rhs_tile`的第一维必须相等，即两者的K维必须一致；K必须为64的倍数。
- scale Tile的逻辑shape必须与配对数据Tile的MX分组一致：`scale_a.shape == [M,K/32]`，`scale_b.shape == [K/32,N]`。该约束按逻辑shape校验，不要使用Tile分配字节数代替。
- scale Tile与数据Tile必须满足硬件地址绑定：

  ```text
  addr(scale_a) = addr(lhs_tile) >> 4
  addr(scale_b) = addr(rhs_tile) >> 4
  ```

  该关系是强制约束，而非地址规划建议。MX矩阵指令根据Left/Right地址隐式定位scale；scale搬入了其他地址时，该指令仍会读取上述映射地址，导致计算结果错误。地址空间的详细说明见[`MemorySpace`](../../basic_data_structures/MemorySpace.md)。

GM中的scale使用普通`DT_FP8E8M0` ND Tensor，并直接声明打包后的实际shape，例如`[M,G/2,2]`或`[G/2,N,2]`。框架不识别或校验group/phase轴，只在`load`目标为fractal-32 ZZ/NN Mat Tile时，根据目标和`order`选择PTO MX layout。MXFP4尾数在GM中以两个4 bit元素打包为一个字节，因此ND Tensor的最后一维是逻辑元素数的一半；搬入后的L1/L0 Tile shape仍使用逻辑shape。FP4主数据场景仍可通过`set_stride`设置按逻辑FP4元素计算的stride。

## 调用示例

### MXFP8示例

下面计算一个不分K块的MXFP8矩阵乘法。A scale的逻辑shape为`[128,4]`，B scale为`[4,128]`，因为每个scale覆盖32个K方向元素。

```python
import pypto_pro.language as pl

TILE = 128
SCALE_K = TILE // 32


@pl.jit(auto_mutex=True)
def mxfp8_matmul_kernel(
    a: pl.Tensor[[TILE, TILE], pl.DT_FP8E4M3FN],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP8E5M2],
    scale_a: pl.Tensor[[TILE, SCALE_K // 2, 2], pl.DT_FP8E8M0],
    scale_b: pl.Tensor[[SCALE_K // 2, TILE, 2], pl.DT_FP8E8M0],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP8E4M3FN,
                         target_memory=pl.MemorySpace.Mat,
                         layout=pl.NZ),
        addrs=0x00000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP8E5M2,
                         target_memory=pl.MemorySpace.Mat,
                         layout=pl.NZ),
        addrs=0x10000, mutex_ids=[1])
    sa_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, SCALE_K], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.Mat,
                         layout=pl.ZZ),
        addrs=0x20000, mutex_ids=[2])
    sb_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[SCALE_K, TILE], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.Mat,
                         layout=pl.NN),
        addrs=0x21000, mutex_ids=[3])
    a_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP8E4M3FN,
                         target_memory=pl.MemorySpace.Left,
                         layout=pl.NZ),
        addrs=0x0000, mutex_ids=[4])
    b_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP8E5M2,
                         target_memory=pl.MemorySpace.Right,
                         layout=pl.ZN),
        addrs=0x0000, mutex_ids=[5])
    sa_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, SCALE_K], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.ScaleLeft,
                         layout=pl.ZZ),
        addrs=0x0000, mutex_ids=[6])
    sb_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[SCALE_K, TILE], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.ScaleRight,
                         layout=pl.NN),
        addrs=0x0000, mutex_ids=[7])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc,
                         layout=pl.NZ, fractal=1024),
        addrs=0x0000, mutex_ids=[8])

    with pl.section_cube():
        al1, bl1 = a_l1.current(), b_l1.current()
        sal1, sbl1 = sa_l1.current(), sb_l1.current()
        al0, bl0 = a_l0.current(), b_l0.current()
        sal0, sbl0 = sa_l0.current(), sb_l0.current()
        ac = acc.current()
        pl.load(al1, a, [0, 0])
        pl.load(bl1, b, [0, 0])
        pl.load(sal1, scale_a, [0, 0, 0], order=[0, 1])
        pl.load(sbl1, scale_b, [0, 0, 0], order=[0, 1])
        pl.move(al0, al1)
        pl.move(bl0, bl1)
        pl.move(sal0, sal1)
        pl.move(sbl0, sbl1)
        pl.matmul_mx(ac, al0, bl0, sal0, sbl0)
        pl.store(out, ac, [0, 0])
```

### MXFP4示例

MXFP4在GM中使用紧凑的4 bit存储：最后一维相邻两个逻辑元素打包为一个字节，第一个元素放在低4 bit，第二个元素放在高4 bit。因此，逻辑shape为`[M,K]`和`[K,N]`的A、B，在Tensor标注中分别写成`[M,K/2]`和`[K,N/2]`。L1及L0 Tile仍使用未压缩的逻辑shape，并通过`set_stride`把Tensor stride设置为按逻辑FP4元素计数。

下面使用E2M1左矩阵和E1M2右矩阵，计算一个MXFP4交叉格式矩阵乘：

```python
import pypto_pro.language as pl

TILE = 128
SCALE_K = TILE // 32


@pl.jit(auto_mutex=True)
def mxfp4_matmul_kernel(
    a: pl.Tensor[[TILE, TILE // 2], pl.DT_FP4E2M1],
    b: pl.Tensor[[TILE, TILE // 2], pl.DT_FP4E1M2],
    scale_a: pl.Tensor[[TILE, SCALE_K // 2, 2], pl.DT_FP8E8M0],
    scale_b: pl.Tensor[[SCALE_K // 2, TILE, 2], pl.DT_FP8E8M0],
    out: pl.Tensor[[TILE, TILE], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP4E2M1,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x00000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP4E1M2,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x10000, mutex_ids=[1])
    sa_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, SCALE_K], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZZ),
        addrs=0x20000, mutex_ids=[2])
    sb_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[SCALE_K, TILE], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NN),
        addrs=0x21000, mutex_ids=[3])
    a_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP4E2M1,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0000, mutex_ids=[4])
    b_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP4E1M2,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0000, mutex_ids=[5])
    sa_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, SCALE_K], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.ScaleLeft, layout=pl.ZZ),
        addrs=0x0000, mutex_ids=[6])
    sb_l0 = pl.make_tile_group(
        type=pl.TileType(shape=[SCALE_K, TILE], dtype=pl.DT_FP8E8M0,
                         target_memory=pl.MemorySpace.ScaleRight, layout=pl.NN),
        addrs=0x0000, mutex_ids=[7])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024),
        addrs=0x0000, mutex_ids=[8])

    with pl.section_cube():
        # Tensor shape按打包字节声明，stride按逻辑FP4元素设置。
        pl.set_stride(a, [TILE, 1])
        pl.set_stride(b, [TILE, 1])

        al1, bl1 = a_l1.current(), b_l1.current()
        sal1, sbl1 = sa_l1.current(), sb_l1.current()
        al0, bl0 = a_l0.current(), b_l0.current()
        sal0, sbl0 = sa_l0.current(), sb_l0.current()
        ac = acc.current()
        pl.load(al1, a, [0, 0])
        pl.load(bl1, b, [0, 0])
        pl.load(sal1, scale_a, [0, 0, 0], order=[0, 1])
        pl.load(sbl1, scale_b, [0, 0, 0], order=[0, 1])
        pl.move(al0, al1)
        pl.move(bl0, bl1)
        pl.move(sal0, sal1)
        pl.move(sbl0, sbl1)
        pl.matmul_mx(ac, al0, bl0, sal0, sbl0)
        pl.store(out, ac, [0, 0])
```

E2M1 × E2M1、E1M2 × E1M2以及E1M2 × E2M1的写法相同，只需替换A、B Tensor和对应L1/L0 Tile的dtype。scale始终使用`DT_FP8E8M0`。

同时使用多组L0 buffer时，每组scale地址都必须与对应数据地址满足“右移4位”的绑定关系。可参考仓库中的`test_matmul_mx.py`完整用例。
