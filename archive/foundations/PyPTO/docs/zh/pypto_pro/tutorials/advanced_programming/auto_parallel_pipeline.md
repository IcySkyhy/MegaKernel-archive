# 自动CV并行流水

## 功能说明

CV融合算子中，cube和vector的计算相互依赖，若按串行流水执行，一个核工作时另一个核只能空等。为了提升CV之间的并行度，让上游核提前若干次迭代计算，下游核则处理上游已经算好的较早迭代的数据，两个核错开节奏、同时工作，从而把彼此的等待时间掩盖掉。

自动CV并行流水是pypto_pro.language.jit的一项编译期变换，包含两个能力：

- **自动流水排布**：把用户手写的CV融合算子串行流水kernel代码，自动改写成并行流水版本，提升性能；
- **自动核间同步**：用户开发的串行版本不需要手写任何CV核间同步指令，并行流水版本会插入全部所需的核间同步。

下图对比了同一个算子（4个stage，cube/vector交替）在串行流水与并行流水（cube提前2次执行）下的执行图：

![串行流水与并行流水的执行节奏对比](../figures/pro_parallel_pipeline_serial_vs_parallel.png "串行流水与并行流水的执行节奏对比")

图中`sN·iM`表示第N个stage正在处理第M次迭代的数据，格子宽度代表该stage的耗时（各stage耗时不同，图中为示意值）。

需要注意的是，流水化的收益上限由**较慢的那个核**决定：图中vector侧单次迭代的耗时高于cube侧，稳态节奏就由vector侧决定，cube侧会出现等待间隙。stage划分越均衡（两个核的耗时越接近），流水填充得越满，收益越高。此外流水的建立和排空各需要若干拍，迭代次数越多，这部分开销占比越低。

## 使用方法
### 1. 编写stage函数
需要用户将算子划分为若干个计算流程，每个计算流程对应一个stage函数，通过@pypto_pro.language.pipeline.stage装饰器进行标识。
```python
import pypto_pro.language as pl

@pl.pipeline.stage
def stage1(ki, a, b_l1, a_l1_db, left_db, right_db, acc_db, mm1_vec_db):
    """Cube：mm1 = A_i @ B"""
    cur_a = a_l1_db.next()
    pl.load(cur_a, a, [ki * TILE, 0])
    ...  # 普通的 Tile/Buffer 操作，不用写任何同步
```

### 2. 声明跨核共享Buffer
跨核共享Buffer使用make_tile_group接口进行声明，tile数目由用户自主分配，通过fwd_ids和bwd_ids参数配置核间正反向同步id，若未配置，则不会插入对应的核间同步。
```python
import pypto_pro.language as pl

mm1_vec_db = pl.make_tile_group(
    type=pl.TileType(shape=[TILE_HALF, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
    addrs=0x0000,
    mutex_ids=[12, 13],
    fwd_ids=[0, 1],
    bwd_ids=[2, 3],
)
```

### 3. 编写主循环
主循环内stage函数按照依赖关系顺序进行书写，需要保证stage之间为CV交替。
```python
import pypto_pro.language as pl

for ki in pl.range(0, N_ITER):
    with pl.section_cube():
        stage1(ki, a, b_l1, a_l1_db, left_db, right_db, acc_db, mm1_vec_db)
    with pl.section_vector():
        stage2(ki, sub_id, mm1_vec_db, relu_vec_db, relu_nz_db, p_mat_db)
    with pl.section_cube():
        stage3(ki, d_l1, p_mat_db, left_db, right_db, acc_db, out_vec_db)
    with pl.section_vector():
        stage4(ki, sub_id, out, out_vec_db)
```

### 4. 开启流水变换

在pypto_pro.language.jit接口中通过PipelineConfig参数进行配置：

| 参数 | 含义 |
|---|---|
| preload | 上游核提前迭代计算的次数。数值越大，越能把数据搬运、计算的延迟掩盖掉，性能通常更好，但会导致头尾开销增大，可根据实际情况调整 |
| sync_only | True时只插核间同步，不做流水改写，用于验证串行版本精度。

**建议流程**：先配置**sync_only=True**执行，确认串行流水版本精度正确，再开启perload参数。

```python
import pypto_pro.language as pl

@pl.jit(auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(sync_only=True))
def pipeline_demo_kernel(...):
    ...
```

```python
import pypto_pro.language as pl

@pl.jit(auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=2))
def pipeline_demo_kernel(...):
    ...
```

### 5. 查看生成的代码
框架自动生成的并行流水代码保存在编译产物目录下，文件名为pipeline_generated.py，用户可以在该代码的基础上继续修改调试。

## 使用约束
- 所有stage调用需要放在同一个for循环内。
- 暂不支持stage嵌套stage。
- stage函数不支持有返回值。
- 每个with pypto_pro.language.section_cube()/pypto_pro.language.section_vector()块里只放单个stage调用，且stage调用需要严格按cube/vector交替排列（C→V→C→V…），不允许连续两个stage落在同一个核上。
- 允许通过if语句判断stage执行场景，但分支条件必须为编译期常量。
- fwd_ids/bwd_ids取值范围为0~15。
- 允许跨核Buffer之间、跨核Buffer与核内Buffer之间进行地址复用，但最多允许两块Buffer复用，且复用双方的Tile数需要一致。
- 一个跨核Buffer（通过fwd_ids/bwd_ids标记）需要恰好被两个stage使用，且这两个stage分别在cube和vector上，构成一对一的生产者/消费者关系。
- 跨核Buffer的Tiles必须随迭代顺序轮转。
- stage函数如果有结构体参数，请使用pl.struct()声明，不支持pl.struct_array()。
- 跨核Buffer仅支持UB/L1。

## 调用示例
```python
import os

import pypto_pro.language as pl
import pytest
import torch

import pypto

ST_DEVICE_ID = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
ST_DEVICE = f"npu:{ST_DEVICE_ID}"

TILE = 64
TILE_HALF = TILE // 2  # dual mode 沿 M 劈开，每个 AIV 处理一半
N_ITER = 4


@pl.pipeline.stage
def stage1(ki, a, b_l1, a_l1_db, left_db, right_db, acc_db, mm1_vec_db):
    """Cube：mm1 = A_i @ B"""
    cur_a = a_l1_db.next()
    pl.load(cur_a, a, [ki * TILE, 0])
    b_slot = b_l1.current()
    left = left_db.next()
    right = right_db.next()
    acc = acc_db.next()
    pl.move(left, cur_a)
    pl.move(right, b_slot)
    pl.matmul(acc, left, right)
    mm1_vec = mm1_vec_db.next()
    # DualModeSplitM：[TILE, TILE] 的 acc 沿 M 劈开，每个 AIV 拿 [TILE_HALF, TILE]
    pl.move(mm1_vec, acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


@pl.pipeline.stage
def stage2(ki, sub_id, mm1_vec_db, relu_vec_db, relu_nz_db, p_mat_db):
    """Vector：对本 AIV 那半做 relu，转 NZ 后按行偏移 insert 回 L1"""
    mm1_vec = mm1_vec_db.next()
    relu_vec = relu_vec_db.next()
    pl.relu(relu_vec, mm1_vec)
    relu_nz = relu_nz_db.next()
    pl.move(relu_nz, relu_vec)  # ND -> NZ，insert 要求源 tile 为 NZ 格式
    p_mat = p_mat_db.next()
    pl.insert(p_mat, relu_nz, [sub_id * TILE_HALF, 0])


@pl.pipeline.stage
def stage3(ki, d_l1, p_mat_db, left_db, right_db, acc_db, out_vec_db):
    """Cube：mm2 = relu(mm1) @ D"""
    d_slot = d_l1.current()
    p_mat = p_mat_db.next()
    left = left_db.next()
    right = right_db.next()
    acc = acc_db.next()
    pl.move(left, p_mat)
    pl.move(right, d_slot)
    pl.matmul(acc, left, right)
    out_vec = out_vec_db.next()
    pl.move(out_vec, acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


@pl.pipeline.stage
def stage4(ki, sub_id, out, out_vec_db):
    """Vector：每个 AIV 写回本迭代结果的一半"""
    out_vec = out_vec_db.next()
    pl.store(out, out_vec, [ki * TILE + sub_id * TILE_HALF, 0])


@pl.jit(auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=2))
def pipeline_demo_kernel(
    a: pl.Tensor[[N_ITER * TILE, TILE], pl.DT_FP32],
    b: pl.Tensor[[TILE, TILE], pl.DT_FP32],
    d: pl.Tensor[[TILE, TILE], pl.DT_FP32],
    out: pl.Tensor[[N_ITER * TILE, TILE], pl.DT_FP32],
):
    # ===== 跨核共享 Buffer：声明 fwd_ids/bwd_ids，框架据此自动插同步 =====
    mm1_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_HALF, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[12, 13], fwd_ids=[0, 1], bwd_ids=[2, 3],
    )
    p_mat_db = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
        addrs=0x8000, mutex_ids=[14, 15], fwd_ids=[4, 5], bwd_ids=[6, 7],
    )
    out_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TILE_HALF, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=0x10000, mutex_ids=[16, 17], fwd_ids=[8, 9], bwd_ids=[10, 11],
    )

    # ===== Cube 侧局部 Buffer =====
    with pl.section_cube():
        a_l1_db = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0x0000, mutex_ids=[0, 1],
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0x10000, mutex_ids=[2],
        )
        d_l1 = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Mat, layout=pl.NZ),
            addrs=0x14000, mutex_ids=[3],
        )
        left_db = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Left, layout=pl.NZ),
            addrs=0x0000, mutex_ids=[4, 5],
        )
        right_db = pl.make_tile_group(
            type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Right, layout=pl.ZN),
            addrs=0x0000, mutex_ids=[6, 7],
        )
        acc_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE, TILE], dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024,
            ),
            addrs=0x0000, mutex_ids=[8, 9, 10, 11],
        )
        # B、D 在循环外一次搬入，全程复用
        b_slot = b_l1.current()
        d_slot = d_l1.current()
        pl.load(b_slot, b, [0, 0])
        pl.load(d_slot, d, [0, 0])

    # ===== Vector 侧局部 Buffer =====
    with pl.section_vector():
        sub_id = pl.get_subblock_idx()
        relu_vec_db = pl.make_tile_group(
            type=pl.TileType(shape=[TILE_HALF, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=0x8000, mutex_ids=[18, 19],
        )
        relu_nz_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TILE_HALF, TILE], dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Vec, layout=pl.NZ,
            ),
            addrs=0x18000, mutex_ids=[20, 21],
        )

    # ===== 流水循环：按 cube/vector 交替调用 4 个 stage =====
    for ki in pl.range(0, N_ITER):
        with pl.section_cube():
            stage1(ki, a, b_l1, a_l1_db, left_db, right_db, acc_db, mm1_vec_db)
        with pl.section_vector():
            stage2(ki, sub_id, mm1_vec_db, relu_vec_db, relu_nz_db, p_mat_db)
        with pl.section_cube():
            stage3(ki, d_l1, p_mat_db, left_db, right_db, acc_db, out_vec_db)
        with pl.section_vector():
            stage4(ki, sub_id, out, out_vec_db)


@pytest.mark.soc("950")
@pypto.options(pass_options={"enable_slice": False})
def test_pipeline_demo_kernel():
    device = ST_DEVICE
    torch.npu.set_device(device)
    torch.manual_seed(0)
    a = torch.randn(N_ITER * TILE, TILE, device=device, dtype=torch.float32)
    b = torch.randn(TILE, TILE, device=device, dtype=torch.float32)
    d = torch.randn(TILE, TILE, device=device, dtype=torch.float32)
    out = torch.zeros(N_ITER * TILE, TILE, device=device, dtype=torch.float32)

    pipeline_demo_kernel(a, b, d, out)
    torch.npu.synchronize()

    ref = torch.relu(a @ b) @ d
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
```
