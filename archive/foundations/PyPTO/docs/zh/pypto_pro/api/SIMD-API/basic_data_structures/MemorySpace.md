# pypto_pro.language.MemorySpace

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

内存空间枚举，用于标记Tile所在的物理存储位置，是[`pypto_pro.language.TileType`](TileType.md)的关键属性。

不同内存空间对应昇腾芯片上不同的物理存储区域，决定了Tile能参与哪些计算、需要哪条流水搬运。

## 取值

| 取值 | 物理位置 | 说明 | 典型用途 |
|---|---|---|---|
| `pypto_pro.language.MemorySpace.DDR` | 片外DDR | 全局内存，Tensor所在 | GM张量存储 |
| `pypto_pro.language.MemorySpace.Vec` | 片上UB | 向量/统一缓冲区 | 向量计算（element-wise、reduce等）的输入输出 |
| `pypto_pro.language.MemorySpace.Mat` | 片上L1 | 矩阵缓冲区 | matmul的L1暂存（GM→L1→L0A/L0B两跳的中间站） |
| `pypto_pro.language.MemorySpace.Left` | 片上L0A | 左操作数缓冲区 | matmul左矩阵输入 |
| `pypto_pro.language.MemorySpace.Right` | 片上L0B | 右操作数缓冲区 | matmul右矩阵输入 |
| `pypto_pro.language.MemorySpace.Acc` | 片上L0C | 累加器缓冲区 | matmul累加器输出 |
| `pypto_pro.language.MemorySpace.Scaling` | 片上 | 缩放/量化参数缓冲区 | quantization/反量化参数 |
| `pypto_pro.language.MemorySpace.ScaleLeft` | 片上L0A（scale专用地址域） | A矩阵的E8M0 scale（分组缩放因子）缓冲区 | `matmul_mx`的`scale_a` |
| `pypto_pro.language.MemorySpace.ScaleRight` | 片上L0B（scale专用地址域） | B矩阵的E8M0 scale（分组缩放因子）缓冲区 | `matmul_mx`的`scale_b` |
| `pypto_pro.language.MemorySpace.Bias` | Bias Buffer | 底层偏置缓冲区标识 | 当前CCE Tile codegen未实现该内存空间映射，不能用于`make_tile`或`make_tile_group`创建Tile |

## 补充说明

不同内存空间的tile在构造[`pypto_pro.language.TileType`](TileType.md)时有不同的默认`layout`：

| 内存空间 | A3默认`layout` | A5默认`layout` | 额外允许 |
|---|---|---|---|
| `Vec` | 无约束 | 无约束 | — |
| `Mat` | `pl.NZ` | `pl.NZ` | `pl.ZN`转置分形布局；UINT64/INT64还允许`pl.ND` |
| `Left` | `pl.ZZ` | `pl.NZ` | 同时允许`pl.ZZ`和`pl.NZ` |
| `Right` | `pl.ZN` | `pl.ZN` | — |
| `Acc` | `pl.NZ` | `pl.NZ` | FP32/INT32自动`fractal=1024` |
| `Scaling` | `pl.ND` | `pl.ND` | — |
| `ScaleLeft` | — | `pl.ZZ` | 仅用于A矩阵的`DT_FP8E8M0`分组缩放因子，32字节地址对齐 |
| `ScaleRight` | — | `pl.NN` | 仅用于B矩阵的`DT_FP8E8M0`分组缩放因子，32字节地址对齐 |

### MX scale（MX矩阵计算的E8M0分组缩放因子）配套缓冲区

`ScaleLeft`和`ScaleRight`分别是与L0A和L0B配套的MX scale缓冲区，在PTO-ISA中分别使用`__ca__`和`__cb__`地址空间。它们具有独立于Left/Right数据地址域的4KB逻辑容量，并非从普通L0A/L0B的64KB数据空间中划出；scale Tile的起始地址由配对数据Tile的起始地址推导：

```text
ScaleLeftAddr  = LeftAddr  >> 4
ScaleRightAddr = RightAddr >> 4
```

因此，64KB的L0A/L0B数据地址范围对应4KB的ScaleLeft/ScaleRight地址范围。例如，Left起始地址为`0x8000`时，配对的ScaleLeft起始地址为`0x0800`。该映射是MX矩阵指令的强制寻址约束。对于显式指定且编译期可知的Tile地址，框架会校验ScaleLeft/ScaleRight与Left/Right的地址映射；自动分配或动态地址无法在该阶段校验。映射不一致时，指令会从由Left/Right地址推导的位置读取scale，而不是从错误配置的scale Tile地址读取，导致计算结果错误。

> [!NOTE]说明
> 4KB的ScaleLeft/ScaleRight不是从64KB的L0A/L0B数据容量中划出的预留空间。使用ScaleLeft/ScaleRight时，Left/Right的数据地址域仍可使用完整64KB，不需要缩减为60KB。

这里的`/16`来自硬件地址右移4位，与“一个E8M0 scale对应K方向连续32个尾数元素”的数据分组规则不是同一概念。MX矩阵乘法的完整参数约束见[`matmul_mx`](../operation/matrix_computation/matmul_mx.md)。

## 调用示例

```python
import pypto_pro.language as pl
# UB tile（向量计算）
tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)

# L1 tile（matmul 中间暂存）
tt_l1 = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat)

# L0A tile（matmul 左矩阵）
tt_left = pl.TileType(shape=[128, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left,
                       layout=pl.NZ)

# Acc tile（matmul 累加器）
tt_acc = pl.TileType(shape=[128, 128], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc)
```
