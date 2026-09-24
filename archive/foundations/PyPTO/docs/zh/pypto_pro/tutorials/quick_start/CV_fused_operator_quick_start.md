# CV融合算子快速入门

本示例是一个入门实践，基于PyPTO Pro SIMD实现Matmul+Softmax融合算子，帮助您快速上手Cube与Vector流水线协同的融合算子开发。它完整呈现了矩阵乘（Cube流水）与行向Softmax（Vector流水）在同一Kernel中的协作流程，助您建立多流水融合的整体认知。开始前，请先参考[环境准备](../../../install/prepare_environment.md)完成基础环境搭建。

## Matmul+Softmax融合算子

**功能介绍**：该融合算子将矩阵乘法与Softmax激活融合到一个Kernel中，数学表达式为 $out = \text{Softmax}(A \times B)$。其中矩阵乘法 $S = A \times B$ 在Cube流水线上执行，Softmax按行归一化在Vector流水线上执行，中间结果通过Global Memory的workspace中转，Cube与Vector之间通过[`set_cross_core`/`wait_cross_core`](../../api/SIMD-API/operation/synchronization/set_cross_core_wait_cross_core.md)进行流水线同步。

- Softmax按行归一化的数学表达式为：

  $$
  out_{i,j} = \frac{e^{S_{i,j} - m_i}}{\sum_{j} e^{S_{i,j} - m_i}}, \qquad m_i = \max_{j} S_{i,j}
  $$

- 矩阵维度：$A \in \mathbb{R}^{M \times 128}$，$B \in \mathbb{R}^{128 \times N}$，$S = A \times B \in \mathbb{R}^{M \times N}$。M/N使用运行时动态维度（`pl.DYNAMIC`）；本入门示例在一个Cube Tile内处理M，支持`1 <= M <= 64`。K固定为128，每次Matmul完整计算K轴；N轴按64分块，N可以包含多个分块，例如1024或4096。Cube直接计算S并写入形状为$[M,N]$的workspace，Vector以$[M半块,N块]$视图沿N方向分块完成Softmax。

## 算子设计

| 模块 | 说明 |
|:---|:---|
| Kernel函数定义 | 通过`@pl.jit(auto_mutex=True)`声明JIT编译目标，开启Tile mutex自动管理；M/N为`pl.DYNAMIC`、K固定为128，本示例限定M不超过64行 |
| Cube Tile定义 | 使用`pl.TileType`定义Mat/Left/Right/Acc上的Tile，FP16输入、FP32累加；Cube直接按$A\times B$计算S；Left/Right/Acc设置`compact=1`以处理M/N尾块 |
| N轴分块 | N分$\lceil N/64 \rceil$块写入workspace；每个N块使用完整的K = 128输入执行一次`pl.matmul` |
| 流水线同步 | Cube段由FIX流水通过`set_cross_core`发出跨核信号，Vector段由MTE2流水通过`wait_cross_core`等待信号后执行load |
| Vector Tile定义 | Vec Tile形状为`[32, 64]`；两个AIV subblock分别处理最多32个M行，并用`set_validshape`设置当前M半块和N块的有效区域；归约结果使用`[M半块, 1]`的DN视图广播，并通过同一UB地址上的`[1, M半块]`行主序视图完成逐元素合并 |
| Vector计算 | 第一遍用`maximum(..., dim=0)`计算各N块的行最大值，再用逐元素`maximum`合并块间结果；第二遍用`exp`和`sum(..., dim=0)`计算各块的指数和，再用逐元素`add`累加完整分母；第三遍用`expand_div(..., dim=0)`完成归一化 |
| 数据搬出 | 归一化结果保持`[M半块, N块]`形状，通过`pl.store`直接写回GM |

## 算子代码实现

### Tile API版本

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

TILE_M = 64
TILE_N = 64
VEC_ROWS = 32
K_SIZE = 128

@pl.jit(auto_mutex=True)
def matmul_softmax_kernel(a: pl.Tensor[[pl.DYNAMIC, K_SIZE], pl.DT_FP16],
                          b: pl.Tensor[[K_SIZE, pl.DYNAMIC], pl.DT_FP16],
                          out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
                          workspace: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32]):
    M = a.shape[0]
    N = b.shape[1]
    N_TILES = (N + TILE_N - 1) // TILE_N
    valid_m = pl.min(TILE_M, M)

    # ---- Cube Tile：直接计算 QK = A @ B ----
    tt_a_mat = pl.TileType(shape=[TILE_M, K_SIZE], dtype=pl.DT_FP16,
                           target_memory=pl.MemorySpace.Mat,
                           valid_shape=[-1, -1])
    tt_b_mat = pl.TileType(shape=[K_SIZE, TILE_N], dtype=pl.DT_FP16,
                           target_memory=pl.MemorySpace.Mat,
                           valid_shape=[-1, -1])
    tt_left = pl.TileType(shape=[TILE_M, K_SIZE], dtype=pl.DT_FP16,
                          target_memory=pl.MemorySpace.Left,
                          valid_shape=[-1, -1], compact=1)
    tt_right = pl.TileType(shape=[K_SIZE, TILE_N], dtype=pl.DT_FP16,
                           target_memory=pl.MemorySpace.Right,
                           valid_shape=[-1, -1], compact=1)
    tt_acc = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc,
                         valid_shape=[-1, -1], compact=1)

    a_l1 = pl.make_tile_group(type=tt_a_mat, addrs=0x0000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(type=tt_b_mat, addrs=0x4000, mutex_ids=[1])
    a_l0a = pl.make_tile_group(type=tt_left, addrs=0x0000, mutex_ids=[2])
    b_l0b = pl.make_tile_group(type=tt_right, addrs=0x0000, mutex_ids=[3])
    qk_l0c = pl.make_tile_group(type=tt_acc, addrs=0x0000, mutex_ids=[4])

    # ---- Vector Tile：每个 AIV 处理最多 32 个 M 行，沿 N 块分三遍完成 Softmax ----
    tt_vec = pl.TileType(shape=[VEC_ROWS, TILE_N], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])
    tt_red = pl.TileType(shape=[VEC_ROWS, 1], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec, layout=pl.DN,
                         valid_shape=[-1, -1])
    # 行归约使用 [M半块, 1] 的 DN 视图，逐元素合并使用同一内存的行主序视图。
    tt_red_rm = pl.TileType(shape=[1, VEC_ROWS], dtype=pl.DT_FP32,
                            target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1])

    qk_vec = pl.make_tile_group(type=tt_vec, addrs=0x0000, mutex_ids=[5])
    tmp_vec = pl.make_tile_group(type=tt_vec, addrs=0x2000, mutex_ids=[6])
    exp_vec = pl.make_tile_group(type=tt_vec, addrs=0x4000, mutex_ids=[7])
    red_vec = pl.make_tile_group(type=tt_red, addrs=0x6000, mutex_ids=[8])
    red_rm = pl.make_tile_group(type=tt_red_rm, addrs=0x6000, mutex_ids=[9])
    global_max = pl.make_tile_group(type=tt_red, addrs=0x6100, mutex_ids=[10])
    global_max_rm = pl.make_tile_group(type=tt_red_rm, addrs=0x6100, mutex_ids=[11])
    global_sum = pl.make_tile_group(type=tt_red, addrs=0x6200, mutex_ids=[12])
    global_sum_rm = pl.make_tile_group(type=tt_red_rm, addrs=0x6200, mutex_ids=[13])

    # ==== Cube: workspace = QK，N 轴分块 ====
    with pl.section_cube():
        cur_a_l1 = a_l1.current()
        cur_b_l1 = b_l1.current()
        cur_a_l0a = a_l0a.current()
        cur_b_l0b = b_l0b.current()
        cur_qk_l0c = qk_l0c.current()
        for nj in pl.range(0, N_TILES, 1):
            n_off = nj * TILE_N
            valid_n = pl.min(TILE_N, N - n_off)
            pl.set_validshape(cur_qk_l0c, [valid_m, valid_n])
            pl.set_validshape(cur_a_l1, [valid_m, K_SIZE])
            pl.set_validshape(cur_b_l1, [K_SIZE, valid_n])
            pl.load(cur_a_l1, a, [0, 0])
            pl.load(cur_b_l1, b, [0, n_off])
            pl.set_validshape(cur_a_l0a, [valid_m, K_SIZE])
            pl.set_validshape(cur_b_l0b, [K_SIZE, valid_n])
            pl.move(cur_a_l0a, cur_a_l1)
            pl.move(cur_b_l0b, cur_b_l1)
            pl.matmul(cur_qk_l0c, cur_a_l0a, cur_b_l0b)
            pl.store(workspace, cur_qk_l0c, [0, n_off])
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    # ==== Vector: 在 QK 上沿 N 方向分块计算 Softmax ====
    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)
        m_off = sub_id * VEC_ROWS
        if m_off < M:
            valid_rows = pl.min(VEC_ROWS, M - m_off)
            cur_qk = qk_vec.current()
            cur_tmp = tmp_vec.current()
            cur_exp = exp_vec.current()
            cur_red = red_vec.current()
            cur_red_rm = red_rm.current()
            cur_global_max = global_max.current()
            cur_global_max_rm = global_max_rm.current()
            cur_global_sum = global_sum.current()
            cur_global_sum_rm = global_sum_rm.current()
            pl.set_validshape(cur_red, [valid_rows, 1])
            pl.set_validshape(cur_red_rm, [1, valid_rows])
            pl.set_validshape(cur_global_max, [valid_rows, 1])
            pl.set_validshape(cur_global_max_rm, [1, valid_rows])
            pl.set_validshape(cur_global_sum, [valid_rows, 1])
            pl.set_validshape(cur_global_sum_rm, [1, valid_rows])

            # 第一遍：合并所有 N 块的行最大值。
            for nj in pl.range(0, N_TILES, 1):
                n_off = nj * TILE_N
                valid_n = pl.min(TILE_N, N - n_off)
                pl.set_validshape(cur_qk, [valid_rows, valid_n])
                pl.set_validshape(cur_tmp, [valid_rows, valid_n])
                pl.load(cur_qk, workspace, [m_off, n_off])
                pl.maximum(cur_red, cur_qk, cur_tmp, dim=0)
                if nj == 0:
                    pl.mul(cur_global_max_rm, cur_red_rm, 1.0)
                else:
                    pl.maximum(cur_global_max_rm, cur_global_max_rm, cur_red_rm)

            # 第二遍：基于全局最大值累加所有 N 块的 exp 和。
            for nj in pl.range(0, N_TILES, 1):
                n_off = nj * TILE_N
                valid_n = pl.min(TILE_N, N - n_off)
                pl.set_validshape(cur_qk, [valid_rows, valid_n])
                pl.set_validshape(cur_tmp, [valid_rows, valid_n])
                pl.set_validshape(cur_exp, [valid_rows, valid_n])
                pl.load(cur_qk, workspace, [m_off, n_off])
                pl.expand_sub(cur_exp, cur_qk, cur_global_max, dim=0)
                pl.exp(cur_exp, cur_exp)
                pl.sum(cur_red, cur_exp, cur_tmp, dim=0)
                if nj == 0:
                    pl.mul(cur_global_sum_rm, cur_red_rm, 1.0)
                else:
                    pl.add(cur_global_sum_rm, cur_global_sum_rm, cur_red_rm)

            # 第三遍：逐块归一化并直接写回 out。
            for nj in pl.range(0, N_TILES, 1):
                n_off = nj * TILE_N
                valid_n = pl.min(TILE_N, N - n_off)
                pl.set_validshape(cur_qk, [valid_rows, valid_n])
                pl.set_validshape(cur_exp, [valid_rows, valid_n])
                pl.load(cur_qk, workspace, [m_off, n_off])
                pl.expand_sub(cur_exp, cur_qk, cur_global_max, dim=0)
                pl.exp(cur_exp, cur_exp)
                pl.expand_div(cur_exp, cur_exp, cur_global_sum, dim=0)
                pl.store(out, cur_exp, [m_off, n_off])


# Host端调用
device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
device = f"npu:{device_id}"
torch.npu.set_device(device)
torch.manual_seed(0)
M, N = 48, 1024
a = torch.randn(M, K_SIZE, device=device, dtype=torch.float16) * 0.1
b = torch.randn(K_SIZE, N, device=device, dtype=torch.float16) * 0.1
out = torch.zeros(M, N, device=device, dtype=torch.float32)
workspace = torch.zeros(M, N, device=device, dtype=torch.float32)

matmul_softmax_kernel(a, b, out, workspace)
torch.npu.synchronize()

matmul_golden = torch.matmul(a.float(), b.float())
torch.testing.assert_close(workspace, matmul_golden, rtol=1e-2, atol=1e-2)
golden = torch.softmax(matmul_golden, dim=-1)
torch.testing.assert_close(out, golden, rtol=1e-2, atol=1e-2)
print("Matmul-Softmax kernel passed!")
```

> [!NOTE]说明
>
> - 该融合算子在同一个Kernel内同时使用`section_cube()`和`section_vector()`，Cube流水执行矩阵乘法，Vector流水执行Softmax，体现了AI Core多流水线协同计算的能力。
> - M/N为`pl.DYNAMIC`，GM维度在运行时通过`a.shape[0]`和`b.shape[1]`获取；K固定为128。本示例在一个Cube Tile内处理M，支持`1 <= M <= 64`；N按64分块，循环次数由运行时N计算，因此同一Kernel可处理1024、4096等不同N值。
> - Cube段沿N轴分块。每个N块完整载入K = 128的A/B子块并执行一次`pl.matmul`；L0C在矩阵乘前设置`valid_shape([valid_m, valid_n])`，使M/N尾块使用当前有效尺寸。
> - `valid_shape=[-1, -1]`在`TileType`中声明Tile的有效区域为运行时动态，后续通过`pl.set_validshape`在运行时设置。Left/Right/Acc上的`compact=1`使L1到L0的分形重排和L0C到GM的存储按当前有效尺寸使用紧凑步长，用于处理M/N不整除Tile尺寸的尾块。
> - Cube直接计算$A\times B$，workspace与输出保持相同的`[M, N]`形状。Cube与Vector之间通过`set_cross_core`/`wait_cross_core`同步，确保workspace写入完成后Vector才开始读取。
> - Vector段通过`pl.get_subblock_idx()`获取AIV subblock编号：subblock 0处理M轴前32行，subblock 1处理后32行；M尾块通过`set_validshape([valid_rows, valid_n])`收窄。
> - Vector不把完整N轴放入一个Tile，而是按64列读取S。第一遍得到完整N轴最大值，第二遍得到基于该最大值的完整N轴分母，第三遍逐块归一化并直接写回。`[M半块, N块]`视图沿`dim=0`归约后得到`[M半块, 1]`的DN结果，供`expand_sub`和`expand_div`按行广播；跨N块的最大值与分母通过同一UB地址上的`[1, M半块]`行主序别名执行`maximum`和`add`，不需要额外搬运。
> - 调用示例将标准正态随机输入缩小到0.1倍，避免未缩放点积使Softmax进入高度饱和区，从而放大NPU计算结果与PyTorch参考结果之间的浮点舍入差异；该缩放仅用于测试数据，不属于Kernel计算。
> - `auto_mutex=True`自动管理各Tile的mutex锁，开发者无需手写`mutex_lock`/`mutex_unlock`。
> - 如需进一步了解PyPTO Pro的SIMD编程模型，请参阅[编程范式概述](../programming_paradigm/programming_paradigm_overview.md)。

### 基于VF（Vector Function）的寄存器级实现

VF（Vector Function，向量函数）是PyPTO Pro中用于描述寄存器级向量计算的函数。VF函数使用`@pl.vector_function`定义，并通过[`vf.*`接口](../../api/SIMD-API/operation/vf_computation/index.md)将UB中的数据加载到向量寄存器，完成计算后再写回UB。外层Kernel仍使用Tile API负责Cube计算、GM与UB之间的数据搬运、UB转置和流水同步。

本节保持输入、输出以及三遍Softmax算法不变，仅将Vector段的Softmax计算改为VF实现。当前`[M半块,N块]`先通过`pl.transpose`转换为UB中的`[N块,64]`物理块，使VF寄存器的每个FP32 lane对应一个M行；随后遍历N轴数据，逐lane更新该行的最大值、指数和及归一化结果。

首次接触VF编程时，建议先阅读[Reg矢量计算编程](../operator_development/tile_based_python_programming/Reg_vector_computation.md)。

| 模块 | 说明 |
|:---|:---|
| VF数据组织 | `qk_nd`从`[M,N]` workspace载入当前`[M半块,N块]`，`pl.transpose`将其转为`qk_dn[N块,64]`；每个N位置占一个64-lane FP32寄存器，`valid_rows`生成的predicate只使能当前AIV负责的M行 |
| VF状态 | `global_max[1,64]`和`global_sum[1,64]`在UB中保存完整N轴对应的逐行状态；第一遍更新最大值，第二遍累加`exp(x-max)`，第三遍归一化 |
| 尾块处理 | M尾块由`vf.update_mask(valid_rows)`限定有效lane；N尾块由VF循环上界`valid_n`限定有效N位置；转回`[M半块,N块]`后按`valid_shape`写回GM |
| VF内存顺序 | `vf.store_align`写入的状态会被后续`vf.load_align`读取，使用`vf.mem_bar(mode=pl.MemBarMode.VST_VLD)`保证VF store到后续Vector load的顺序；MTE与Vector之间的依赖仍由Tile mutex管理 |

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

TILE_M = 64
TILE_N = 64
VEC_ROWS = 32
K_SIZE = 128
VF_LANES = 64
NEG_INF = -1e30

@pl.vector_function
def softmax_vf_init(global_max, global_sum, valid_rows: pl.DT_INT64):
    """初始化跨 N 块保存的逐行最大值与指数和。"""
    preg = vf.update_mask(valid_rows, dtype=pl.DT_FP32)
    max_reg = vf.full(NEG_INF, preg, dtype=pl.DT_FP32)
    sum_reg = vf.full(0.0, preg, dtype=pl.DT_FP32)
    vf.store_align(global_max, max_reg, preg)
    vf.store_align(global_sum, sum_reg, preg)


@pl.vector_function
def softmax_vf_update_max(src_dn, global_max,
                          valid_rows: pl.DT_INT64, valid_n: pl.DT_INT64):
    """把当前 N 块合并到逐行全局最大值。"""
    preg = vf.update_mask(valid_rows, dtype=pl.DT_FP32)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    max_reg = vf.load_align(global_max, 0)
    for ni in pl.range(0, valid_n):
        src_reg = vf.load_align(src_dn, ni * VF_LANES)
        max_reg = vf.max(max_reg, src_reg, preg)
    vf.store_align(global_max, max_reg, preg)


@pl.vector_function
def softmax_vf_update_sum(src_dn, global_max, global_sum,
                          valid_rows: pl.DT_INT64, valid_n: pl.DT_INT64):
    """基于全局最大值累加当前 N 块的指数和。"""
    preg = vf.update_mask(valid_rows, dtype=pl.DT_FP32)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    max_reg = vf.load_align(global_max, 0)
    sum_reg = vf.load_align(global_sum, 0)
    for ni in pl.range(0, valid_n):
        src_reg = vf.load_align(src_dn, ni * VF_LANES)
        exp_reg = vf.exp_sub(src_reg, max_reg, preg)
        sum_reg = vf.add(sum_reg, exp_reg, preg)
    vf.store_align(global_sum, sum_reg, preg)


@pl.vector_function
def softmax_vf_normalize(src_dn, dst_dn, global_max, global_sum,
                         valid_rows: pl.DT_INT64, valid_n: pl.DT_INT64):
    """使用完整 N 轴的最大值与指数和归一化当前 N 块。"""
    preg = vf.update_mask(valid_rows, dtype=pl.DT_FP32)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    max_reg = vf.load_align(global_max, 0)
    sum_reg = vf.load_align(global_sum, 0)
    for ni in pl.range(0, valid_n):
        src_reg = vf.load_align(src_dn, ni * VF_LANES)
        exp_reg = vf.exp_sub(src_reg, max_reg, preg)
        out_reg = vf.div(exp_reg, sum_reg, preg)
        vf.store_align(dst_dn + ni * VF_LANES, out_reg, preg)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.jit(auto_mutex=True)
def matmul_softmax_vf_kernel(a: pl.Tensor[[pl.DYNAMIC, K_SIZE], pl.DT_FP16],
                             b: pl.Tensor[[K_SIZE, pl.DYNAMIC], pl.DT_FP16],
                             out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
                             workspace: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32]):
    M = a.shape[0]
    N = b.shape[1]
    N_TILES = (N + TILE_N - 1) // TILE_N
    valid_m = pl.min(TILE_M, M)

    # ---- Cube Tile：直接计算 QK = A @ B ----
    tt_a_mat = pl.TileType(shape=[TILE_M, K_SIZE], dtype=pl.DT_FP16,
                           target_memory=pl.MemorySpace.Mat,
                           valid_shape=[-1, -1])
    tt_b_mat = pl.TileType(shape=[K_SIZE, TILE_N], dtype=pl.DT_FP16,
                           target_memory=pl.MemorySpace.Mat,
                           valid_shape=[-1, -1])
    tt_left = pl.TileType(shape=[TILE_M, K_SIZE], dtype=pl.DT_FP16,
                          target_memory=pl.MemorySpace.Left,
                          valid_shape=[-1, -1], compact=1)
    tt_right = pl.TileType(shape=[K_SIZE, TILE_N], dtype=pl.DT_FP16,
                           target_memory=pl.MemorySpace.Right,
                           valid_shape=[-1, -1], compact=1)
    tt_acc = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc,
                         valid_shape=[-1, -1], compact=1)

    a_l1 = pl.make_tile_group(type=tt_a_mat, addrs=0x0000, mutex_ids=[0])
    b_l1 = pl.make_tile_group(type=tt_b_mat, addrs=0x4000, mutex_ids=[1])
    a_l0a = pl.make_tile_group(type=tt_left, addrs=0x0000, mutex_ids=[2])
    b_l0b = pl.make_tile_group(type=tt_right, addrs=0x0000, mutex_ids=[3])
    qk_l0c = pl.make_tile_group(type=tt_acc, addrs=0x0000, mutex_ids=[4])

    # ---- Vector Tile：TTRANS 前后都使用 64x64 物理 Tile，VF 用 mask 收窄 M 尾块 ----
    tt_vf_block = pl.TileType(shape=[VF_LANES, TILE_N], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Vec,
                              valid_shape=[-1, -1])
    tt_vf_state = pl.TileType(shape=[1, VF_LANES], dtype=pl.DT_FP32,
                              target_memory=pl.MemorySpace.Vec,
                              valid_shape=[-1, -1])

    qk_nd = pl.make_tile_group(type=tt_vf_block, addrs=0x0000, mutex_ids=[5])
    qk_dn = pl.make_tile_group(type=tt_vf_block, addrs=0x4000, mutex_ids=[6])
    out_dn = pl.make_tile_group(type=tt_vf_block, addrs=0x8000, mutex_ids=[7])
    out_nd = pl.make_tile_group(type=tt_vf_block, addrs=0xC000, mutex_ids=[8])
    global_max = pl.make_tile_group(type=tt_vf_state, addrs=0x10000, mutex_ids=[9])
    global_sum = pl.make_tile_group(type=tt_vf_state, addrs=0x10100, mutex_ids=[10])

    # ==== Cube: workspace = QK，N 轴分块 ====
    with pl.section_cube():
        cur_a_l1 = a_l1.current()
        cur_b_l1 = b_l1.current()
        cur_a_l0a = a_l0a.current()
        cur_b_l0b = b_l0b.current()
        cur_qk_l0c = qk_l0c.current()
        for nj in pl.range(0, N_TILES, 1):
            n_off = nj * TILE_N
            valid_n = pl.min(TILE_N, N - n_off)
            pl.set_validshape(cur_qk_l0c, [valid_m, valid_n])
            pl.set_validshape(cur_a_l1, [valid_m, K_SIZE])
            pl.set_validshape(cur_b_l1, [K_SIZE, valid_n])
            pl.load(cur_a_l1, a, [0, 0])
            pl.load(cur_b_l1, b, [0, n_off])
            pl.set_validshape(cur_a_l0a, [valid_m, K_SIZE])
            pl.set_validshape(cur_b_l0b, [K_SIZE, valid_n])
            pl.move(cur_a_l0a, cur_a_l1)
            pl.move(cur_b_l0b, cur_b_l1)
            pl.matmul(cur_qk_l0c, cur_a_l0a, cur_b_l0b)
            pl.store(workspace, cur_qk_l0c, [0, n_off])
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)

    # ==== Vector: 在转置后的 [N块,64] UB Tile 上调用 VF ====
    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        pl.system.wait_cross_core(pipe=pl.PipeType.MTE2, event_id=0)
        m_off = sub_id * VEC_ROWS
        if m_off < M:
            valid_rows = pl.min(VEC_ROWS, M - m_off)
            cur_qk_nd = qk_nd.current()
            cur_qk_dn = qk_dn.current()
            cur_out_dn = out_dn.current()
            cur_out_nd = out_nd.current()
            cur_global_max = global_max.current()
            cur_global_sum = global_sum.current()
            pl.set_validshape(cur_global_max, [1, valid_rows])
            pl.set_validshape(cur_global_sum, [1, valid_rows])
            softmax_vf_init(cur_global_max, cur_global_sum, valid_rows)

            # 第一遍：把各 N 块转为 [N块,64]，逐 lane 更新每行最大值。
            for nj in pl.range(0, N_TILES, 1):
                n_off = nj * TILE_N
                valid_n = pl.min(TILE_N, N - n_off)
                pl.set_validshape(cur_qk_nd, [valid_rows, valid_n])
                pl.set_validshape(cur_qk_dn, [valid_n, valid_rows])
                pl.load(cur_qk_nd, workspace, [m_off, n_off])
                pl.transpose(cur_qk_dn, cur_qk_nd)
                softmax_vf_update_max(cur_qk_dn, cur_global_max, valid_rows, valid_n)

            # 第二遍：基于全局最大值逐 lane 累加完整 N 轴的指数和。
            for nj in pl.range(0, N_TILES, 1):
                n_off = nj * TILE_N
                valid_n = pl.min(TILE_N, N - n_off)
                pl.set_validshape(cur_qk_nd, [valid_rows, valid_n])
                pl.set_validshape(cur_qk_dn, [valid_n, valid_rows])
                pl.load(cur_qk_nd, workspace, [m_off, n_off])
                pl.transpose(cur_qk_dn, cur_qk_nd)
                softmax_vf_update_sum(cur_qk_dn, cur_global_max, cur_global_sum,
                                      valid_rows, valid_n)

            # 第三遍：VF 归一化后转回 [M半块,N块] 并写回 GM。
            for nj in pl.range(0, N_TILES, 1):
                n_off = nj * TILE_N
                valid_n = pl.min(TILE_N, N - n_off)
                pl.set_validshape(cur_qk_nd, [valid_rows, valid_n])
                pl.set_validshape(cur_qk_dn, [valid_n, valid_rows])
                pl.set_validshape(cur_out_dn, [valid_n, valid_rows])
                pl.set_validshape(cur_out_nd, [valid_rows, valid_n])
                pl.load(cur_qk_nd, workspace, [m_off, n_off])
                pl.transpose(cur_qk_dn, cur_qk_nd)
                softmax_vf_normalize(cur_qk_dn, cur_out_dn, cur_global_max,
                                     cur_global_sum, valid_rows, valid_n)
                pl.transpose(cur_out_nd, cur_out_dn)
                pl.store(out, cur_out_nd, [m_off, n_off])


# Host端调用
device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
device = f"npu:{device_id}"
torch.npu.set_device(device)
torch.manual_seed(0)
M, N = 48, 1024
a = torch.randn(M, K_SIZE, device=device, dtype=torch.float16) * 0.1
b = torch.randn(K_SIZE, N, device=device, dtype=torch.float16) * 0.1
out = torch.zeros(M, N, device=device, dtype=torch.float32)
workspace = torch.zeros(M, N, device=device, dtype=torch.float32)

matmul_softmax_vf_kernel(a, b, out, workspace)
torch.npu.synchronize()

matmul_golden = torch.matmul(a.float(), b.float())
torch.testing.assert_close(workspace, matmul_golden, rtol=1e-2, atol=1e-2)
golden = torch.softmax(matmul_golden, dim=-1)
torch.testing.assert_close(out, golden, rtol=1e-2, atol=1e-2)
print("Matmul-Softmax VF kernel passed!")
```

> [!NOTE]说明
>
> - VF版本没有改变算子的数学方向：输入、workspace和输出在GM中仍为`[M,N]`，Softmax仍沿N轴计算。`[N块,64]`只是为了让VF寄存器lane对应M行而在UB内使用的转置视图。
> - FP32 VF寄存器包含64个lane。每个AIV最多处理32个M行，`vf.update_mask(valid_rows)`只使能对应lane；保留64列物理宽度可使每个N位置的起始地址按一个完整寄存器对齐。
> - 转置前后的Vec Tile都声明为64×64，用于保持FP32 VF寄存器所需的64元素行步长；`TTRANS`的实际转置范围取自源Tile当前的`valid_shape`。代码将源、目标的有效区域分别设置为`[valid_rows, valid_n]`和`[valid_n, valid_rows]`，VF再通过循环上界和predicate处理N/M尾块。
> - `vf.mem_bar(mode=pl.MemBarMode.VST_VLD)`只用于存在VF/Vector store后续又被Vector load读取的局部内存依赖，包括跨N块读取`global_max`/`global_sum`以及归一化结果被`pl.transpose`读取；MTE2、MTE3与VF之间的Tile流水依赖仍由`auto_mutex=True`管理。
