# pypto_pro.language.make_tile_group

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

一次性创建一组**同规格、可轮转复用**的Tile，为每块Tile绑定独立地址，并可选建立Tile与一个或多个`mutex_id`的对应关系。解析器会将group展开成多个[`make_tile`](make_tile.md)；配合`auto_mutex`，框架根据该元数据为涉及这些Tile的后续操作自动插入mutex同步，是实现多缓冲（N-buffer）流水并行的核心接口。

缓冲深度由`depth`或非空`mutex_ids`的长度确定：

- **1-buffer**（`mutex_ids`长度1）：单块tile，用`current()`取用，无轮转
- **2-buffer**（double-buffer / ping-pong）：两块tile交替使用，最常用
- **3-buffer / 4-buffer / N-buffer**：更多tile轮转，适用于搬运延迟较高、需要更深流水掩盖的场景

与多次调用[`pypto_pro.language.make_tile`](make_tile.md)相比，`make_tile_group`的优势是：统一管理地址布局、轮转游标和mutex元数据，无需手动维护Tile下标与同步关系。group句柄本身不在运行时执行加锁/解锁；其mutex映射由`auto_mutex`在编译期消费。

下图以UB中的双缓冲为例，展示单个基地址如何展开成两个连续Tile槽位，以及地址、mutex和轮转访问之间的对应关系。

![make_tile_group双缓冲地址与轮转关系](../../../figures/make_tile_group_rotation.jpg "make_tile_group双缓冲地址与轮转关系")

## 函数原型

```python
pypto_pro.language.make_tile_group(type=, addrs=, mutex_ids=None, depth=None, fwd_ids=None, bwd_ids=None) -> group
```


## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `type` | 输入 | `pypto_pro.language.TileType`，组内每块tile的统一规格 |
| `addrs` | 输入 | 单个基地址（组内tile连续排布），或地址列表（逐块指定） |
| `mutex_ids` | 输入 | 可选的mutex ID配置，可为`None`、空列表，或非空整数列表/元组。配置非空时，框架为每块Tile记录一个或多个ID：例如`[1, 2, 3, 4]`为4块Tile各分配一个ID，`[[1, 2], [3, 4]]`为2块Tile各分配两个ID。配置为`None`或空列表时，group只管理Tile、地址和轮转游标，不参与`auto_mutex`，跨pipe依赖需由用户保证 |
| `depth` | 输入 | 可选的Tile数量；`mutex_ids`为`None`或空列表时必填 |
| `fwd_ids` | 输入 | 可选的跨核前向event_id列表，用于自动Pipeline变换中生产者→消费者的跨核同步。启用`pipeline=pl.pipeline.PipelineConfig(...)`后，框架在写入该缓冲的stage末尾自动插入`set_cross_core(event_id=fwd_ids[i])`，在读取该缓冲的stage开头自动插入`wait_cross_core(event_id=fwd_ids[i])` |
| `bwd_ids` | 输入 | 可选的跨核后向event_id列表，用于自动Pipeline变换中消费者→生产者的反向同步（通知生产者缓冲已释放）。框架在读取stage末尾插入`set_cross_core(event_id=bwd_ids[i])`，在写入stage开头插入`wait_cross_core(event_id=bwd_ids[i])` |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `type` | 输入 | 须为[`pypto_pro.language.TileType`](../../basic_data_structures/TileType.md)；非该类型报错 |
| `addrs` | 输入 | 给单个基址时，第i块地址自动取`base + i × slot_size`（`slot_size`为单块字节数）；给列表时，长度必须等于`mutex_ids`长度或者确定的`depth`，逐块指定（用于非连续布局） |
| `mutex_ids` | 输入 | 可为`None`、空列表或整数列表。整数取值范围`[0, 31]`；每块Tile的ID数量必须一致，且同一Tile内多个ID时ID不得重复，不同Tile间ID可重复 |
| `depth` | 输入 | 正的编译期整数。非空`mutex_ids`未指定时由其长度推导；两者同时指定时必须相等 |
| `fwd_ids` | 输入 | 可选的非空整数列表或元组，元素为`set_cross_core`/`wait_cross_core`的event_id（取值0～15）。列表长度须与缓冲深度（`depth`或`mutex_ids`长度）一致，用于按轮转索引`i % N`选择event_id。仅在启用`pipeline`变换时生效；未启用时忽略 |
| `bwd_ids` | 输入 | 同`fwd_ids`，用于反向同步。可单独使用（单方向通知）或与`fwd_ids`成对使用（双向握手） |

## 补充说明

为MX多缓冲创建`ScaleLeft`/`ScaleRight` group时，scale地址列表必须与配对数据地址列表逐项绑定：

```text
scale_left_addrs[i]  = left_addrs[i]  >> 4
scale_right_addrs[i] = right_addrs[i] >> 4
```

例如，`Left` group的`addrs=[0x0000, 0x8000]`时，配对`ScaleLeft` group必须使用`addrs=[0x0000, 0x0800]`。不满足该关系时，scale会被搬到MX矩阵指令不会为该数据Tile读取的地址，导致计算结果错误。

返回一个group句柄，提供三个方法和下标形式取出裸Tile。配置非空`mutex_ids`时，Tile与mutex的映射会随返回值保留，供`auto_mutex`使用：

| 方法 | 说明 |
|---|---|
| `group.next()` | 推进轮转状态并返回下一个Tile；连续调用时按组大小N循环选择槽位 |
| `group.current()` | 不推进轮转状态，返回当前选择的Tile |
| `group.previous()` | 不推进轮转状态，返回当前选择之前的Tile |
| `group[i]` | 按照下标进行索引取第i个Tile，不读取也不推进轮转状态 |

使用`group[i]`进行下标索引时，`i`的取值范围为`[0, num_tile)`。`mutex_ids`非空时，`num_tile`为`len(mutex_ids)`；否则为确定的`depth`。

group句柄可直接传给[`pypto_pro.language.set_validshape`](../memory_vector_computation/transpose_and_element_access/set_validshape.md)，对group中所有tile批量设置valid_shape，适用于全局只需设置一次、后续直接`next()`的场景。

配置`fwd_ids`/`bwd_ids`后，该group成为跨核共享缓冲，用于Cube与Vector之间的自动Pipeline同步。框架根据stage对该缓冲的读写角色自动插入`set_cross_core`/`wait_cross_core`：

- **生产者stage（写）**：开头`wait_cross_core(bwd_ids)`等待消费者释放，末尾`set_cross_core(fwd_ids)`通知消费者数据就绪
- **消费者stage（读）**：开头`wait_cross_core(fwd_ids)`等待生产者数据就绪，末尾`set_cross_core(bwd_ids)`通知生产者缓冲已释放

`fwd_ids`和`bwd_ids`的列表长度应与缓冲深度一致，框架按轮转索引`i % N`选择当次迭代使用的event_id。仅在`@pl.jit(pipeline=pl.pipeline.PipelineConfig(...))`启用自动Pipeline变换时生效。

## 调用示例

### 2-buffer（double-buffer / ping-pong）

`make_tile_group` + `auto_mutex`的典型场景是matmul：L1暂存用`next()`轮转开双缓冲，L0A/L0B/L0C用单mutex ID的group配`current()`。开启`auto_mutex=True`后，框架按Tile的mutex元数据为涉及这些Tile的搬运和计算自动插入互斥操作。对于不带mutex元数据的其他Tile或额外数据依赖，仍需要用户显式同步。

```python
import pypto_pro.language as pl

TILE = 128
MM_M, MM_K, MM_N = 256, 128, 256


@pl.jit(auto_mutex=True)
def tile_group_matmul_kernel(
    a: pl.Tensor[[MM_M, MM_K], pl.DT_FP16],
    b: pl.Tensor[[MM_K, MM_N], pl.DT_FP16],
    c: pl.Tensor[[MM_M, MM_N], pl.DT_FP32],
):
    # L1 双缓冲（next() 轮转）
    a_l1_db = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, MM_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1])
    b_l1_db = pl.make_tile_group(
        type=pl.TileType(shape=[MM_K, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x10000, mutex_ids=[2, 3])
    # L0A / L0B / Acc 单 tile group（current()）
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, MM_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[4])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[MM_K, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[5])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[6])

    with pl.section_cube():
        for i in pl.range(0, MM_M, TILE):
            for j in pl.range(0, MM_N, TILE):
                cur_a = a_l1_db.next()      # 双缓冲轮转
                cur_b = b_l1_db.next()
                al = a_left.current()       # 单 tile group
                br = b_right.current()
                ac = acc.current()
                pl.load(cur_a, a, [i, 0])
                pl.load(cur_b, b, [0, j])
                pl.move(al, cur_a)
                pl.move(br, cur_b)
                pl.matmul(ac, al, br)
                pl.store(c, ac, [i, j])
```

### 4-buffer

L1使用4块tile轮转，适用于搬运延迟较高（如大K维分块）的场景。`mutex_ids`长度为4，`next()`按`cursor % 4`自动轮转。

```python
import pypto_pro.language as pl

TILE = 128
MM_M, MM_K, MM_N = 256, 128, 256


@pl.jit(auto_mutex=True)
def tile_group_4buf_matmul_kernel(
    a: pl.Tensor[[MM_M, MM_K], pl.DT_FP16],
    b: pl.Tensor[[MM_K, MM_N], pl.DT_FP16],
    c: pl.Tensor[[MM_M, MM_N], pl.DT_FP32],
):
    a_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, MM_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x00000, mutex_ids=[0, 1, 2, 3])
    b_l1 = pl.make_tile_group(
        type=pl.TileType(shape=[MM_K, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat),
        addrs=0x20000, mutex_ids=[4, 5, 6, 7])
    a_left = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, MM_K], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left),
        addrs=0x0000, mutex_ids=[8])
    b_right = pl.make_tile_group(
        type=pl.TileType(shape=[MM_K, TILE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right),
        addrs=0x0000, mutex_ids=[9])
    acc = pl.make_tile_group(
        type=pl.TileType(shape=[TILE, TILE], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc),
        addrs=0x0000, mutex_ids=[10])

    with pl.section_cube():
        for i in pl.range(0, MM_M, TILE):
            for j in pl.range(0, MM_N, TILE):
                cur_a = a_l1.next()
                cur_b = b_l1.next()
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

非连续布局可用地址列表逐块指定：

```python
# 两块 tile 分别落在 0x0 和 0x10000
tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
buf = pl.make_tile_group(type=tt, addrs=[0x0, 0x10000], mutex_ids=[0, 1])
```
