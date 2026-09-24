# pypto_pro.language.load

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

把GM中一块数据按**绝对元素坐标**搬入L1/UB Tile，是Kernel获取输入数据并参与计算的基础接口。GM数据支持在行（Row）、列（Col）维度上偏移任意元素个数，支持连续搬运和高维切分搬运两种模式。

如果希望按“第几块Tile”来定位（自动乘以Tile形状），需要使用[`pypto_pro.language.load_tile`](load_tile.md)。

## 函数原型

```python
pypto_pro.language.load(dst_tile, src_tensor, offsets, *, order=None)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_tile` | 输出 | 只能是L1、UB Tile，搬入目的地 |
| `src_tensor` | 输入 | Tensor类型，来自GM的源数据 |
| `offsets` | 输入 | GM地址偏移，标记在每根轴上的初始偏移，单位元素个数 |
| `order` | 输入 | 可选，Tile维度在GlobalTensor维度中对应哪几根轴；元素为Tensor绝对轴索引，升序表示不转置，反序表示转置；省略时默认`[ndim-2, ndim-1]`（不转置） |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_tile` | 输出 | 数据类型：b8、b16、b32、b64<br>尾块处理：<br>• 可通过set_validshape设置尾块大小，Tile shape需要32字节对齐，不对齐报错<br>• valid_shape可以不对齐<br>• Vec ND尾块不需要配置compact；涉及紧凑排列、分形转换或Cube计算时，按对应数据路径配置compact = 1，详见[`CompactMode`](../../basic_data_structures/CompactMode.md)和[尾块处理](../../../../tutorials/operator_development/tile_based_python_programming/tail_block_handling.md)<br>• 支持设定padding值<br>地址配置：<br>• 作为`load`目的Tile时，只支持`Mat`（L1）和`Vec`（UB）；`Left`（L0A）、`Right`（L0B）、`Acc`（L0C）、`Scaling`、`ScaleLeft`、`ScaleRight`等其他Tile内存空间由`move`、`matmul`或`store`等对应接口使用<br>• Cube侧目的Tile必须为`Mat`（L1），Vector侧目的Tile必须为`Vec`（UB）；MX scale的E8M0 Tile先load到`Mat`（L1）的ZZ/NN layout Tile，再通过`move`搬入`ScaleLeft`/`ScaleRight`<br>• L1、UB缓冲区首地址必须32字节对齐，不对齐编译报错 |
| `src_tensor` | 输入 | 数据类型：b8、b16、b32、b64<br>layout：支持`ND`、`DN`、`NZ`<br>stride：支持配置stride，stride维度需要等于Tensor维度数，默认不配置时是尾轴stride = 1的连续场景 |
| `offsets` | 输入 | 单位元素个数，大小不超过对应维度的shape，不支持负数索引 |
| `order` | 输入 | 只支持配置Tensor维度范围内的dim，只支持二维数组配置，其余配置报错<br>用于高维Tensor中指定Tile对应哪几个维度；order中轴索引的顺序决定是否转置：升序不转置（ND行主序），反序转置（DN列主序），需要配合Tensor的layout以及Tile的shape和stride填写<br>E8M0搬入fractal-32 ZZ/NN Tile时分别选择A/B scale layout，框架不从shape判断group/phase轴；最后一轴固定作为物理phase轴，不能在`order`中选择<br>省略时，普通Tensor默认取最后两维`[ndim-2, ndim-1]`；带物理尾轴的MX scale默认取尾轴前两维`[ndim-3, ndim-2]`，其他多维结构应显式指定两个矩阵轴 |

## 流水类型

MTE2（GM → L1/UB的搬入流水）。

## 约束说明

当前`DT_FP8E8M0` Tensor搬入`fractal=32`的`ZZ`/`NN` Mat（L1）Tile，仅支持作为`matmul_mx`/`matmul_mx_acc`的scale搬运。普通E8M0数据不支持使用该目标组合；满足该组合的`load`会按MX scale解释，并要求源Tensor的最后一轴是长度为2的物理phase轴。

开启`auto_mutex`时，若连续两次`pl.load`向同一个UB（或L1）Tile地址搬运数据，并且前一次搬入的数据没有被读取，则必须在两次`load`之间调用`pl.system.bar_mte2()`，再复用该地址。

关于复用Tile地址的完整同步规则，请参考下文“Tile地址复用与流水同步”。

## 调用示例

下面是一个完整Kernel：从GM载入两个64×64的输入到UB，相加后写回GM。`pypto_pro.language.load`负责把GM数据搬入UB Tile。Vector Kernel开启`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl

@pl.jit(auto_mutex=True)
def add_kernel(
    a: pl.Tensor[[64, 64], pl.DT_FP16],
    b: pl.Tensor[[64, 64], pl.DT_FP16],
    out: pl.Tensor[[64, 64], pl.DT_FP16],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])

    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.add(cur_out, cur_a, cur_b)
        pl.store(out, cur_out, [0, 0])
```

其他典型用法（节选）：

```python
# 循环按行块载入（matmul 取左矩阵）
for i in pl.range(0, m_dim, 64):
    pl.load(tile_a, a, [i, 0])

# 列主序载入（DN 布局，flash attention 把 K 载入 L1）
pl.load(k_mat, k, [skv_off, 0], order=[1, 0])

# 高维Tensor + order：指定Tile对应Tensor的轴
pl.load(q_buf, q, [b_idx, 0, n_idx, 0], order=[1, 3])
```

## 高级用法

以下介绍`pypto_pro.language.load`在流水同步、复杂数据布局和边界处理场景中的用法。

### Tile地址复用与流水同步

一次`pl.load`会通过MTE2向目标Tile写入数据。开启`auto_mutex`并复用同一个UB（或L1）Tile地址时，可按以下步骤判断是否需要手动同步。

#### 判断步骤

1. 先确认相邻两次`load`的目标是否为**同一个Tile（同一个物理地址）**。若目标地址不同，例如双缓冲或多缓冲轮转到不同Tile，则不适用本节规则。
2. 若目标地址相同，检查第二次`load`前是否已有操作读取前一次搬入的Tile：
   - **已读取**：例如UB上由Vector（V）流水执行的计算（`pl.add`、`pl.mul`），或L1→L0的MTE1 `move`。数据依赖会建立MTE2→V或MTE2→MTE1同步，**无需额外调用`pl.system.bar_mte2()`**。
   - **未读取**：在两次`load`之间调用`pl.system.bar_mte2()`，使前一次MTE2搬运完成后再复用该地址，避免两次写入同一地址的次序或覆盖风险（WAW，Write-After-Write）。与该Tile无关的计算或搬运不属于“读取”。

#### `auto_mutex`与`bar_mte2`的边界

- `auto_mutex`只能处理具有相同mutex ID的Tile在不同Pipeline之间的同步。同一个Tile被反复写入属于同一条Pipeline，框架不会自动补充同步，需要用户按上述步骤处理。
- `pl.system.bar_mte2()`只约束MTE2流水，不会保留随后被覆盖的旧数据。若后续仍需要前一次搬入的数据，必须在复用该地址前先读取或复制该数据。

#### 代码示例

**场景一：复用同一地址前未读取前一次结果——需要`bar_mte2`**

未同步示例（两次MTE2写入同一地址之间没有同步）：

```python
# 同一Tile被连续写入，中间没有读取操作或bar_mte2
pl.load(in_tile, src_a, [0, 0])
pl.load(in_tile, src_b, [64, 0])    # 前一次MTE2写入可能尚未完成

# 部分迭代不读取in_tile，但下一轮仍会复用该地址
for x in pl.range(0, m_dim, 64):
    pl.load(in_tile, src, [x, 0])
    if x % 2 == 0:
        pl.add(out_tile, in_tile, in_tile)   # 偶数轮读取in_tile
    # 奇数轮没有读取in_tile，也没有bar_mte2
```

同步示例（在未读取结果的分支中插入`bar_mte2`）：

```python
# 复用同一Tile前先完成前一次MTE2搬运
pl.load(in_tile, src_a, [0, 0])
pl.system.bar_mte2()
pl.load(in_tile, src_b, [64, 0])

# 没有读取in_tile的分支显式约束MTE2
for x in pl.range(0, m_dim, 64):
    pl.load(in_tile, src, [x, 0])
    if x % 2 == 0:
        pl.add(out_tile, in_tile, in_tile)   # 数据依赖建立MTE2→V同步
    else:
        pl.system.bar_mte2()                  # 下一轮复用in_tile前完成MTE2
```

> 循环写成`load → Vector计算（读取该Tile）→ store`时，每一轮都会在下一次`load`前读取该Tile，因此不需要`bar_mte2`。只有某个分支跳过了这次读取、而后续仍要复用同一Tile时，才需要在该分支中补充`bar_mte2`。

**场景二：复用同一地址前已读取前一次结果——无需`bar_mte2`**

冗余同步示例（重复同步会降低流水并行度）：

```python
# 后续Vector计算读取in_tile，无需额外调用bar_mte2
pl.load(in_tile, src_a, [0, 0])
pl.system.bar_mte2()
pl.add(out_tile, in_tile, in_tile)
pl.load(in_tile, src_b, [64, 0])
```

数据依赖同步示例（无需额外调用`bar_mte2`）：

```python
# Vector计算读取in_tile，因此可在下一次搬运时复用该地址
pl.load(in_tile, src_a, [0, 0])
pl.add(out_tile, in_tile, in_tile)   # 数据依赖建立MTE2→V同步
pl.load(in_tile, src_b, [64, 0])

# 每一轮都会读取in_tile，下一轮可继续复用它
for x in pl.range(0, m_dim, 64):
    pl.load(in_tile, src, [x, 0])
    pl.add(out_tile, in_tile, in_tile)   # 读取in_tile
    pl.store(out, out_tile, [x, 0])
```

### GM Tensor多维场景

Tensor的维度可能有多维，而Tile只有两维，那么Tensor和Tile之间的拷贝需要指定Tile的两维是Tensor中的哪两维。

这由`pl.load` / `pl.load_tile`的`order`参数（以及`store` / `store_tile`的`order`参数）控制：

- **`offsets`的长度 == Tensor的维数**：每个Tensor轴给一个偏移。
- **`order`**：一个长度为2的列表，元素为Tensor绝对轴索引，指出Tile的两维分别落在Tensor的哪两个轴上。order中轴索引的顺序决定是否转置：升序不转置，反序转置。
  未被`order`选中的轴，用`offsets`里对应的值**定死为一个下标**（相当于在那一维上
  切一刀）。
- **默认值**：不填`order`时，取Tensor的**最后两维**，即
  `order = [ndim-2, ndim-1]`（不转置）。

例：一个4维Tensor `q: [B, N, Sq, D]`，Tile是`[TS, TD]`，想让Tile覆盖
`(Sq, D)`这两维、并在`b`、`n`上各定死一个下标：

```python
q: pl.Tensor[[B, N, Sq, D], pl.DT_FP16]
q_tile = pl.make_tile(pl.TileType(shape=[TS, TD], dtype=pl.DT_FP16,
                                  target_memory=pl.MemorySpace.Mat), addr=0x0, size=...)
# 最后两维恰好是 (Sq, D)，默认 order=[2,3]，可省略：
pl.load(q_tile, q, [b, n, sq_off, 0])
```

若想让Tile覆盖**非相邻**或**非末尾**的两维（例如`(N, D)`，即轴1和轴3），显式给
`order`。例如，对于Tensor `[B, S, N, D]`：

```python
# Tile 的两维 = Tensor 的轴 1 与轴 3；轴 0(b_idx)、轴 2(n_idx) 被定死
pl.load_tile(k_mat_buf[buf_idx], k, [b_idx, ki, n_idx, 0], order=[1, 3])
```

> `order`是**编译期常量列表**（它决定代码生成时Tensor视图的stride），不能是运行时
> 变量。

### 尾块需要padding场景

当GM Tensor的shape不能被Tile整除时，边界上会出现比Tile小的**尾块**。Tile的
物理大小固定，但每个尾块的**有效区域**不同。处理方式：

1. Tile的`valid_shape`声明成动态（`[-1, -1]`），load前用`pl.set_validshape`写入这一
   块真实的有效行/列 —— load会**只搬有效区**，不会越界读GM，也不会把padding写回。
2. 若后续算子会读到有效区**之外**（归约 / matmul会整块读入），再配合`pad` +
   `pl.fillpad`把无效区填成安全值（求和填`zero`、`row_max`/softmax填`min`、`row_min`
   填`max`）。

```python
# Tile 物理 64x128，有效行/列运行时决定
tile_type = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16,
                        target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
tile = pl.make_tile(tile_type, addr=0x0, size=64*128*2)
with pl.section_vector():
    # 每个尾块先写有效形状，再 load —— 只搬 rows x cols，不越界
    pl.set_validshape(tile, [rows, cols])
    pl.load(tile, a, [row_off, col_off])
    ...
    pl.store(out, tile, [row_off, col_off])   # 只写回有效区
```

尾块的`valid_shape` / `set_validshape` / `pad` / `fillpad` / `compact`如何协同，详见
[尾块处理](../../../../tutorials/operator_development/tile_based_python_programming/tail_block_handling.md)。

### Cube侧转置场景

matmul的左/右矩阵在从L1（`Mat`）搬到L0（`Left`/`Right`）时可能需要转置。下面先说明硬件约束，再给出前端的写法。

#### 背景：硬件约束

- 左矩阵进入L0A（`Left`），物理形态固定为`shape=[M, K]`、`layout=pl.NZ`；右矩阵进入L0B（`Right`），固定为`shape=[K, N]`、`layout=pl.ZN`。
- `Mat`搬到L0（`pl.move`，底层是CANN的TMOV）**要求源和目的的物理`[Rows, Cols]`完全一致**；转置只能借助`NZ`/`ZN`的fractal差异实现，不能改变维度本身。
- 物理上`Mat [N, K] NZ`与`Mat [K, N] ZN`是**同一份数据（同一段bytes）**，只是标注方式不同。

因此“逻辑转置”本质上要求`Mat`声明成与`Left`/`Right`一致的物理形态，并通过`layout`与`order`表达转置。

#### 写法：显式`order`

`Mat`的`shape`与对应的`Left`/`Right`**保持一致**，靠`layout`与`order`表达转置。

- **不转置**：左矩阵`Mat`的`layout`与`Left`相同（`pl.NZ`）；右矩阵`Mat`的`layout`与`Right`相反（`pl.NZ`）；`pl.load`正常调用。
- **转置**：左矩阵`Mat`的`layout`与`Left`相反（`pl.ZN`）；右矩阵`Mat`的`layout`与`Right`相同（`pl.ZN`）；`pl.load`增加`order=[1, 0]`（反序转置），框架会把对应的`GlobalTensor`标成`DN`并对调stride。

| 矩阵 | 是否转置 | GM Tensor shape | Mat声明 | load |
| --- | --- | --- | --- | --- |
| 左A[M,K] | 否 | `[M, K]` | `shape=[M, K], layout=pl.NZ` | 正常 |
| 左A[M,K] | 是 | `[K, M]` | `shape=[M, K], layout=pl.ZN` | `order=[1, 0]` |
| 右B[K,N] | 否 | `[K, N]` | `shape=[K, N], layout=pl.NZ` | 正常 |
| 右B[K,N] | 是 | `[N, K]` | `shape=[K, N], layout=pl.ZN` | `order=[1, 0]` |

- **左矩阵不转置，从GM拷贝到L1**

```python
gm = pl.make_tensor(ptr, [M, K])
mat_type = pl.TileType(shape=[M, K], dtype=pl.DT_FP16,
                       target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
mat_0 = pl.make_tile(mat_type, addr=0x0, size=32768)
pl.load(mat_0, gm)                          # 正常 load
left = pl.make_tile(pl.TileType(shape=[M, K], dtype=pl.DT_FP16,
                                target_memory=pl.MemorySpace.Left), addr=0x0, size=32768)
pl.move(left, mat_0)                        # Mat[M,K] -> Left[M,K]，同形，不转置
```

- **左矩阵转置，从GM拷贝到L1**

```python
gm = pl.make_tensor(ptr, [K, M])            # 数据是 A^T=[K, M]
mat_type = pl.TileType(shape=[M, K], dtype=pl.DT_FP16,
                       target_memory=pl.MemorySpace.Mat, layout=pl.ZN)
mat_0 = pl.make_tile(mat_type, addr=0x0, size=32768)
pl.load(mat_0, gm, order=[1, 0])       # 转置 load（order 反序）；框架标 DN 并对调 stride
left = pl.make_tile(pl.TileType(shape=[M, K], dtype=pl.DT_FP16,
                                target_memory=pl.MemorySpace.Left), addr=0x0, size=32768)
pl.move(left, mat_0)                        # Mat[M,K] -> Left[M,K]，同形，move 时 fractal 翻转实现转置
```

- **右矩阵不转置，从GM拷贝到L1**

```python
gm = pl.make_tensor(ptr, [K, N])
mat_type = pl.TileType(shape=[K, N], dtype=pl.DT_FP16,
                       target_memory=pl.MemorySpace.Mat, layout=pl.NZ)
mat_0 = pl.make_tile(mat_type, addr=0x0, size=32768)
pl.load(mat_0, gm)                          # 正常 load
right = pl.make_tile(pl.TileType(shape=[K, N], dtype=pl.DT_FP16,
                                 target_memory=pl.MemorySpace.Right), addr=0x0, size=32768)
pl.move(right, mat_0)                        # Mat[K,N] -> Right[K,N]，同形，不转置
```

- **右矩阵转置，从GM拷贝到L1**

```python
gm = pl.make_tensor(ptr, [N, K])            # 数据是 B^T=[N, K]
mat_type = pl.TileType(shape=[K, N], dtype=pl.DT_FP16,
                       target_memory=pl.MemorySpace.Mat, layout=pl.ZN)
mat_0 = pl.make_tile(mat_type, addr=0x0, size=32768)
pl.load(mat_0, gm, order=[1, 0])       # 转置 load（order 反序）；框架标 DN 并对调 stride
right = pl.make_tile(pl.TileType(shape=[K, N], dtype=pl.DT_FP16,
                                 target_memory=pl.MemorySpace.Right), addr=0x0, size=32768)
pl.move(right, mat_0)                        # Mat[K,N] -> Right[K,N]，同形，move 时 fractal 翻转实现转置
```

#### 数据来自UB（片上产生）时的转置

当左/右矩阵不是从GM直接load，而是由Vector侧产生、经`pl.move`(ND→NZ) + `pl.insert`写入L1时，`Mat`的`layout`与`shape`同样需要与`Left`/`Right`保持一致：

- **不转置**：`Mat`的`layout=pl.NZ`，`shape`与`Left`/`Right`一致；`insert`后`move`同形不转置。
- **转置**：`Mat`的`layout=pl.ZN`，`shape`与`Left`/`Right`一致；`insert`写入ZN物理布局（与源NZ Tile物理等价），`move`时SFractal不匹配触发硬件转置。

> 注意：`pl.insert`写入`ZN` Mat时，源Tile的NZ `[R, C]`物理布局等价于`ZN [C, R]`，因此源Tile shape与Mat shape互为转置时物理数据恰好匹配。需确保`insert`的逻辑边界检查通过（`indexCol + validCol ≤ Mat.Cols`）。

| 矩阵 | 是否转置 | Vector产出数据 | Mat声明 |
| --- | --- | --- | --- |
| 左A[M,K] | 否 | `[M, K]` | `shape=[M, K], layout=pl.NZ` |
| 左A[M,K] | 是 | `[K, M]` (Aᵀ) | `shape=[M, K], layout=pl.ZN` |
| 右B[K,N] | 否 | `[K, N]` | `shape=[K, N], layout=pl.NZ` |
| 右B[K,N] | 是 | `[N, K]` (Bᵀ) | `shape=[K, N], layout=pl.ZN` |

> 约束：`pl.insert`要求`tile.dim0 == mat.dim0`（物理行匹配）；分subcore带列偏移插入时，被转置的矩阵通常需为方阵。

### Tensor视图重建与合轴

`pl.make_tensor`在不搬运数据的情况下，通过shape和stride重建GM Tensor视图，可用于调整rank、shape或stride以及合并连续维度。新Tensor视图复用底层存储，`stride`必须与实际内存排布一致。

- **通过`pl.Ptr`重新生成**：Kernel参数声明成裸指针`pl.Ptr[dtype]`，在函数体里用
  `pl.make_tensor(ptr, shape, stride)`按需要的shape和stride构造视图。shape各维的长度可来自运行时TilingData；视图的rank由`shape`列表长度决定，是编译期固定的。

```python
@pl.jit()
def k(q: pl.Ptr[pl.DT_FP16], tiling: OpTiling):
    # 行优先 => stride = [n*d, d, 1]；shape/stride 全部来自 tiling
    tensor_q = pl.make_tensor(q, [tiling.sq, tiling.n, tiling.d],
                              [tiling.n * tiling.d, tiling.d, 1])
    # 之后 tensor_q 的用法与 pl.Tensor 参数完全相同
```

- **从已有`pl.Tensor`重建**：`pl.make_tensor`的第一个参数可以是已有的`pl.Tensor`，
  新视图复用其底层指针，并设置新的shape、stride及可选的dtype。

**“合轴”指把GM Tensor的两个维度当成一维来处理**。

与前面“多维场景”用`order`在多维里挑两维不同，合轴是把**相邻的两维合并**，让
Tensor的有效维数降下来，从而更自然地对上二维Tile。典型用途：

- FA 的**TND / BSND**布局：把 batch 轴`B`和序列轴`S`合成一个“总 token”轴`B * S`
  （即TND里的`T`），一次load就能跨batch连续取若干行。
- 不同维数的Host输入：把前若干维乘到一起，折叠成Kernel内固定的二维`[M, N]` 视图。

#### 合轴的前提：两维在内存里必须连续

只有当被合并的两维在内存中**首尾相接、没有间隔**时才能合轴。对行优先（ND）的
`[d0, d1, d2]`，其stride是`[d1 * d2, d2, 1]`：

- 合并`d0`与`d1` → 新轴大小`d0 * d1`，stride取里层的`d2`。成立条件：
  `stride(d0) == d1 * stride(d1)`，即`d1 * d2 == d1 * d2`，满足连续性条件。
- 如果`d0`与`d1`之间有padding（`stride(d0) > d1 * stride(d1)`），**不能**合轴 ——
  合并后会把那段padding也当成数据读进来。

#### 使用 `pl.make_tensor` 构造降维视图

合轴通过`pl.make_tensor`合并连续维度并构造低一维视图，再由`load`搬入Tile。

**例1：把`[B, S, D]`合成`[B * S, D]`（静态shape）**

```python
# 原始 GM：q 是 [B, S, D]，行优先连续，stride = [S*D, D, 1]
q: pl.Tensor[[B, S, D], pl.DT_FP16]

# 合轴：B、S 合并成一维 B*S，stride 取里层的 D
q_merged = pl.make_tensor(q, [B * S, D], [D, 1])   # 复用 q 的指针，不搬数据

# 现在 q_merged 是二维 [B*S, D]，直接按 Tile 索引 load
tile = pl.make_tile(pl.TileType(shape=[TS, D], dtype=pl.DT_FP16,
                                target_memory=pl.MemorySpace.Vec), addr=0x0, size=TS*D*2)
pl.load_tile(tile, q_merged, [t, 0])   # 第 t 块 = 合并轴上的第 [t*TS : (t+1)*TS] 行
```

合并轴上的偏移换算：要读原始的第`b`个batch、第`s`行，合并后的行号是
`b * S + s`（行优先的自然展开）。

**例 2：运行时shape —— 把Host输入的前若干维折叠成 `[M, N]`**

Kernel可接收裸指针和TilingData，将Host侧2～4维输入的shape折叠成固定的二维Tensor视图再处理。
`N`是最内维，`M`是其余维的乘积（合轴）：

```python
@pl.jit(auto_mutex=True)
def add_dynrank_kernel(x: pl.Ptr[pl.DT_FP16], y: pl.Ptr[pl.DT_FP16],
                       z: pl.Ptr[pl.DT_FP16], tiling: AddTiling):
    N = tiling.shape[3]
    M = tiling.shape[0] * tiling.shape[1] * tiling.shape[2]   # 前三维合轴成 M
    tensor_x = pl.make_tensor(x, [M, N], [N, 1])              # 折叠成二维 [M, N]
    tensor_y = pl.make_tensor(y, [M, N], [N, 1])
    tensor_z = pl.make_tensor(z, [M, N], [N, 1])
    ...
    pl.load_tile(tile_a, tensor_x, [i, j])                    # 之后就是普通二维 load
```

因为逐元素算子只关心"扁平的元素顺序"，把 `[2, 4, 256, 256]`、`[8, 256, 256]`、
`[512, 512]`都可合轴成`[M, N]`，因此同一份Kernel可处理这些不同的Host输入shape。Kernel内由`make_tensor`构造的视图的rank始终固定为2。

**例3：TND / BSND布局的合轴**

FlashAttention的TND布局本身就是把batch与序列合成一个“总token”轴`T = ΣS_i`。若拿到
的是BSND（`[B, S, N, D]`）且各batch的`S`相同、内存连续，可临时合轴成TND视角：

```python
# BSND -> 把 B、S 合成 T = B*S（N、D 保留），stride 取里层
q_tnd = pl.make_tensor(q, [B * S, N, D], [N * D, D, 1])
# 再用 order 在 [T, N, D] 里挑 (T, D) 两维
pl.load_tile(q_tile, q_tnd, [t_off, n_idx, 0], order=[0, 2])
```

#### 小结

| 步骤 | 做法 |
|------|------|
| 1. 确认可合轴 | 被合并的两维在内存里连续（无padding）。 |
| 2. 重建视图   | `pl.make_tensor(src, 合并后的shape, 合并后的stride)`，stride取里层维的stride。 |
| 3. 正常load  | 合并后的Tensor维数更低，按普通二维 / 多维场景load即可。 |
| 4. 偏移换算   | 合并轴的行号 = `外层下标 * 里层大小 + 里层下标`。 |

> **合轴约束**：被合并的维度必须连续，stride必须与实际内存排布一致。违反上述约束会读取错误位置或Padding区域的数据。排查精度问题时，可改用未合轴的逐维Tensor视图进行对比验证。
