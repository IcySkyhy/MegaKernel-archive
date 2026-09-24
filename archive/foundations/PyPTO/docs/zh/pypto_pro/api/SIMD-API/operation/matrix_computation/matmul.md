# pypto_pro.language.matmul

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

完成一次矩阵乘法`dst_tile = lhs_tile × rhs_tile`，数据通路为L0A(Left) × L0B(Right) → L0C(Acc)。输入矩阵需先搬运到L0A/L0B内存空间（一般经GM → L1 → L0A/L0B两跳），结果写入L0C累加器。

当传入可选参数`bias_tile`时，在矩阵乘法后自动融合bias加法：`dst_tile = lhs_tile × rhs_tile + bias_tile`，bias沿M维行广播（shape `[1, N]`广播到`[M, N]`），在fixpipe阶段完成加法。无需额外调用`pl.add`，减少UB往返。

如果要在已有累加结果上继续累加（K维分块的非首块），使用 [`pypto_pro.language.matmul_acc`](matmul_acc.md)。

## 函数原型

```python
pypto_pro.language.matmul(dst_tile, lhs_tile, rhs_tile, bias_tile=None, *, phase=None)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_tile` | 输出 | 只能是Acc/L0C Tile，存放矩阵乘法结果 |
| `lhs_tile` | 输入 | 只能是L0A/Left Tile，左矩阵 |
| `rhs_tile` | 输入 | 只能是L0B/Right Tile，右矩阵 |
| `bias_tile` | 输入 | 可选，Bias Tile，shape为`[1, N]`，沿M维行广播。<br>仅支持作为第4个位置参数传入，不支持`bias_tile=`关键字传参 |
| `phase` | 输入 | 可选，用于K维分块累加场景 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dst_tile` | 输出 | 数据类型：FP16、BF16、FP32、INT32（累加器精度通常高于输入，如FP16输入对应FP32累加）<br>shape：`[M, N]`<br>地址配置：<br>• 只能是Acc/L0C内存空间，其他空间报错<br>• `layout=pl.NZ`；FP32/INT32累加器需设`fractal`（FP32默认1024）<br>• 支持通过`valid_shape=[-1, -1]` + `set_validshape`设置尾块有效大小 |
| `lhs_tile` | 输入 | 数据类型：FP16、BF16、FP32、INT8、HF8<br>shape：`[M, K]`<br>地址配置：<br>• 只能是L0A/Left内存空间，其他空间报错<br>• A3默认`layout=pl.ZZ`；A5默认`layout=pl.NZ` |
| `rhs_tile` | 输入 | 数据类型：与`lhs_tile`一致<br>shape：`[K, N]`，K维需与`lhs_tile`的K一致<br>地址配置：<br>• 只能是L0B/Right内存空间，其他空间报错<br>• `layout=pl.ZN` |
| `bias_tile` | 输入 | 可选，传入时执行`dst = lhs @ rhs + bias`，仅支持作为第4个位置参数传入（`matmul(dst, lhs, rhs, bias)`），不支持`bias_tile=`关键字传参<br>数据类型：FP32（必须与累加器dtype一致，硬件要求`TileRes::DType == TileBias::DType`）<br>shape：`[1, N]`，N维需与`rhs_tile`的N一致，沿M维行广播到`[M, N]`<br>地址配置：<br>• 只能是Bias内存空间（`pl.MemorySpace.Bias`），其他空间报错<br>• bias tile无法直接从GM加载，需先load到L1(Mat) 再`move`到Bias，`move`时自动完成dtype转换（如FP16→FP32、BF16→FP32） |
| `phase` | 输入 | 可选，K维分块累加时控制fixpipe写回GM的unit_flag：<br>• 不传（默认）：单次乘法，无分块累加<br>• `pl.AccPhase.Partial`：中间累加步，表示后续还有K块<br>• `pl.AccPhase.Final`：最终步，表示K累加结束、可写回GM<br>详见 [`matmul_acc`](matmul_acc.md) 的分块累加用法 |

## 调用示例

### 单次matmul（无bias、无K维分块）

完整kernel计算`C[M,N] = A[M,K] @ B[K,N]`，用`make_tile_group` + `auto_mutex`管理L1/L0A/L0B/L0C缓冲。L1暂存用`next()`轮转开ping-pong双缓冲，L0A/L0B/L0C用单mutex_id的group配`current()`。开启`auto_mutex=True`后，相邻搬运与计算间的同步由框架按tile的mutex自动插入，无需手写`sync_src`/`sync_dst`。

下面是K恰好为一个tile（128）的单次matmul：每个`[i, j]`块一次`matmul`直接写回GM，不涉及K维累加，因此不需要`phase` / `fractal`。

> K维分块累加（`phase` + `fractal` + `set_mm_layout_transform`）见 [`pypto_pro.language.matmul_acc`](matmul_acc.md) 的调用示例。

```python
import pypto_pro.language as pl

TILE = 128
M_SIZE = 256
K_SIZE_MM = 128      # K 恰好一个 tile，无需分块累加
N_SIZE = 256


@pl.jit(auto_mutex=True)
def matmul_kernel(
    a: pl.Tensor[[M_SIZE, K_SIZE_MM], pl.DT_FP16],
    b: pl.Tensor[[K_SIZE_MM, N_SIZE], pl.DT_FP16],
    c: pl.Tensor[[M_SIZE, N_SIZE], pl.DT_FP32],
):
    # L1 双缓冲（next() 轮转）
    a_l1_db = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, K_SIZE_MM], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1_db = pl.make_tile_group(
        type=pl.TileType(shape=[K_SIZE_MM, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])
    # L0A / L0B / Acc 单 tile group（current()）
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, K_SIZE_MM], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[4])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[K_SIZE_MM, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[5])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[6])

    with pl.section_cube():
        for i in pl.range(0, M_SIZE, TILE):          # M 维分块
            for j in pl.range(0, N_SIZE, TILE):      # N 维分块
                cur_a = a_l1_db.next()
                cur_b = b_l1_db.next()
                al = a_left.current()
                br = b_right.current()
                ac = acc.current()
                pl.load(cur_a, a, [i, 0])
                pl.load(cur_b, b, [0, j])
                pl.move(al, cur_a)
                pl.move(br, cur_b)
                pl.matmul(ac, al, br)
                pl.store(c, ac, [i, j])
```

### 带bias的matmul（无K维分块）

传入第4个位置参数（`pl.matmul(ac, al, br, bl)`）即可融合bias加法。bias从GM加载到L1(Mat, FP16)，再`move`到L0B(Bias, FP32)，`move`时自动完成FP16→FP32类型转换。

```python
import pypto_pro.language as pl

TILE = 128
M_SIZE = 256
K_SIZE_MM = 128
N_SIZE = 256


@pl.jit(auto_mutex=True)
def matmul_bias_kernel(
    a: pl.Tensor[[M_SIZE, K_SIZE_MM], pl.DT_FP16],
    b: pl.Tensor[[K_SIZE_MM, N_SIZE], pl.DT_FP16],
    bias: pl.Tensor[[1, N_SIZE], pl.DT_FP16],
    c: pl.Tensor[[M_SIZE, N_SIZE], pl.DT_FP16],
):
    a_l1_db = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, K_SIZE_MM], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1_db = pl.make_tile_group(
        type=pl.TileType(shape=[K_SIZE_MM, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])
    bias_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000, mutex_ids=[4, 5])
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, K_SIZE_MM], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[6])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[K_SIZE_MM, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[7])
    bias_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias),
        addrs=0x0000, mutex_ids=[8])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[9])

    with pl.section_cube():
        for i in pl.range(0, M_SIZE, TILE):
            for j in pl.range(0, N_SIZE, TILE):
                cur_a = a_l1_db.next()
                cur_b = b_l1_db.next()
                cur_bias_l1 = bias_l1.next()
                al = a_left.current()
                br = b_right.current()
                bl = bias_l0b.current()
                ac = acc.current()
                pl.load(cur_a, a, [i, 0])
                pl.load(cur_b, b, [0, j])
                pl.load(cur_bias_l1, bias, [0, j])
                pl.move(al, cur_a)
                pl.move(br, cur_b)
                pl.move(bl, cur_bias_l1)             # L1(FP16) → L0B Bias(FP32)，自动类型转换
                pl.matmul(ac, al, br, bl)             # out = A @ B + bias
                pl.store(c, ac, [i, j])
```

### K维分块累加 + bias（首块matmul(bias) + 后续matmul_acc）

计算`D[M,N] = A[M,K] @ B[K,N] + bias[1,N]`，K=384分为3个tile。首块用带bias的`matmul`覆盖写入`A0@B0+bias`，后续块用 [`matmul_acc`](matmul_acc.md) 累加。

带bias的`matmul`底层硬件将累加器初始化为0后写入结果（`cmatrixInitVal=true`，覆盖而非累加），因此 **只能用于首块**，不能用于中间块或末块。

K分块累加链对正确性的硬性要求：

1. **首块`matmul(bias, phase=Partial)`**：覆盖写入`acc = A0@B0 + bias`，不设unit_flag。
2. **中间块`matmul_acc(phase=Partial)`**：累加`acc += Ai@Bi`，不设unit_flag。
3. **末块`matmul_acc(phase=Final)`**：累加`acc += An@Bn`，设unit_flag=1。
4. **store传`phase=pl.STPhase.Final`**：读unit_flag=1后写回GM。
5. **L0C累加器设`fractal=1024`**（FP32）。
6. **cube段用`set_mm_layout_transform(enabled=True)`开启**，段末`enabled=False`关闭。

```python
import pypto_pro.language as pl

TILE = 128
K_SPLIT = 384     # 分 3 个 TILE 块


@pl.jit(auto_mutex=True)
def matmul_k_split_bias_kernel(
    a: pl.Tensor[[TILE, K_SPLIT], pl.DT_FP16],
    b: pl.Tensor[[K_SPLIT, TILE], pl.DT_FP16],
    bias: pl.Tensor[[1, TILE], pl.DT_FP16],
    c: pl.Tensor[[TILE, TILE], pl.DT_FP16],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])
    bias_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000, mutex_ids=[4, 5])
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left,
                         layout=pl.NZ),
        addrs=0x0000, mutex_ids=[6, 7])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[8, 9])
    bias_l0b = pl.make_tile_group(
        type=pl.TileType(shape=[1, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Bias),
        addrs=0x0000, mutex_ids=[10, 11])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc,
                         fractal=1024),
        addrs=0x0000, mutex_ids=[12])

    with pl.section_cube():
        pl.system.set_mm_layout_transform(enabled=True)
        ac = acc.current()
        for k in pl.range(0, K_SPLIT, TILE):
            cur_a = a_l1.next()
            cur_b = b_l1.next()
            al = a_left.next()
            br = b_right.next()
            pl.load(cur_a, a, [0, k])
            pl.load(cur_b, b, [k, 0])
            pl.move(al, cur_a)
            pl.move(br, cur_b)
            if k == 0:
                # 首块：加载 bias，matmul 覆盖写入 acc = A0@B0 + bias
                bias_l1_tile = bias_l1.next()
                pl.load(bias_l1_tile, bias, [0, 0])
                bl = bias_l0b.next()
                pl.move(bl, bias_l1_tile)
                pl.matmul(ac, al, br, bl, phase=pl.AccPhase.Partial)
            elif k < K_SPLIT - TILE:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)
            else:
                pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Final)
        pl.store(c, ac, [0, 0], phase=pl.STPhase.Final)
        pl.system.set_mm_layout_transform(enabled=False)
```

> **bias tile约束**：
> - dtype必须为 **FP32**（与累加器一致），否则编译报`No supported bias data type`。
> - 内存空间必须为`pl.MemorySpace.Bias`，否则codegen报`Invalid MemorySpace value`。
> - 加载路径为GM → L1(Mat, FP16/BF16) → L0B(Bias, FP32)，`move`时自动完成类型转换。不支持直接从GM load到Bias空间（`load`仅支持Vec/Mat目标）。

> **K维分块约束**：
> - 带bias的`matmul`底层`cmatrixInitVal=true`（覆盖acc），**只能用于首块**，不能用于中间块或末块。
> - 首块`matmul(bias, phase=Partial)`后，后续块必须用 [`matmul_acc`](matmul_acc.md) 累加。
> - `phase`配对规则与不带bias的`matmul` / `matmul_acc`一致，详见 [`phase`](phase.md)。
