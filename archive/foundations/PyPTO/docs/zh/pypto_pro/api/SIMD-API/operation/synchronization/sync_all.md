# pypto_pro.language.system.sync_all

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

在多个参与执行的AIV/AIC核之间建立全局屏障。所有参与核到达`sync_all`后才能继续执行。`HARD`模式使用FFTS硬件信号；`SOFT`模式使用GM共享计数器轮询，需要显式workspace。

`sync_all`是参与核之间的同步，不等同于本AI Core内的`bar_all()`，也不能替代两条具体pipe之间的精细`sync_src`/`sync_dst`依赖。

## 函数原型

```python
pypto_pro.language.system.sync_all(
    workspaces=None,
    *,
    core_type=pl.SyncCoreType.MIX,
    mode=pl.SyncAllMode.HARD,
)
```

## 参数类型与范围

| 参数 | 输入/输出 | 类型、范围与语义 |
|---|---|---|
| `workspaces` | 输入 | `HARD`模式必须为`None`或空列表；传入任何非空workspace会报错。`SOFT`模式必须是非空列表，元素按IR类型识别，组合见下表 |
| `core_type` | 输入 | 必须是`pl.SyncCoreType`枚举：`AIV_ONLY`（仅AIV）/ `AIC_ONLY`（仅AIC）/ `MIX`（AIV + AIC，默认） |
| `mode` | 输入 | `pl.SyncAllMode.HARD`（默认）或`pl.SyncAllMode.SOFT` |

### SOFT模式workspace组合

`workspaces`中的元素按类型区分，顺序不影响识别，但每一类最多只能出现一次。

| `core_type` | 必需workspace | 禁止workspace | 可选元素 |
|---|---|---|---|
| `AIV_ONLY` | 1个GM Tensor + 1个`MemorySpace.Vec`的UB Tile | `MemorySpace.Mat`的L1 Tile | 1个Python `int`或整数Scalar `used_cores` |
| `AIC_ONLY` | 1个GM Tensor + 1个`MemorySpace.Mat`的L1 Tile | `MemorySpace.Vec`的UB Tile | 1个Python `int`或整数Scalar `used_cores` |
| `MIX` | 1个GM Tensor + 1个UB Tile + 1个L1 Tile | 无 | 1个Python `int`或整数Scalar `used_cores` |

GM workspace必须是专用于同步的INT32共享计数器空间，不能与其他数据复用。底层软同步至少会访问一条64字节cache line（16个INT32）；首次启动Kernel前应将共享计数器初始化为0，Kernel执行期间不得由其他逻辑修改。当前PyPTO只校验workspace的IR类型和内存空间，不校验dtype、容量、地址重叠或初始值；用户必须按底层SYNCALL约束分配和初始化workspace。

`used_cores`为SOFT模式的可选整数元素，可使用Python `int`或整数Scalar表达式。不传或传`0`时，底层根据启动配置推导参与者数量；非零值必须与实际到达该屏障的参与者数量一致。当前前端只校验它是整数Scalar，不校验其运行时取值。

## 使用约束

> [!CAUTION]注意
> `sync_all`不得放在运行时循环或运行时分支内。应将它放在对应Cube/Vector段的循环外，并保证只有实际参与同一屏障的核执行，且每个参与者以相同顺序和相同次数到达。错误的`core_type`/`used_cores`或两侧不对称的位置可能导致死锁；前端不会跨控制流证明这些条件。

纯Vector Kernel应使用`AIV_ONLY`，纯Cube Kernel应使用`AIC_ONLY`。只有在AIV和AIC都会到达同一屏障时才使用`MIX`；默认值`MIX`不代表它适用于任意Kernel。

> [!IMPORTANT]重要
> Cube、Vector共存的Kernel使用`MIX`时，Cube侧AIC和Vector侧AIV必须都执行对应的`sync_all`；仅在Cube侧调用会因等待未到达的AIV而超时。与`set_cross_core`/`wait_cross_core`并用时，两侧的MIX屏障必须位于该SET/WAIT对的同一侧；禁止Cube侧先执行`sync_all`再SET、Vector侧先WAIT再执行`sync_all`，否则会形成环形等待。

## 调用示例

下面的纯Vector Kernel展示HARD `AIV_ONLY`屏障的放置方式。各AIV按核号处理互不重叠的行，TileGroup和`auto_mutex`负责核内pipe依赖；`sync_all`位于循环外，使所有参与AIV在本阶段结束后再越过屏障。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def sync_all_kernel(
    x: pl.Tensor[[2048, 64], pl.DT_FP32],
    out: pl.Tensor[[2048, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    input_tiles = pl.make_tile_group(
        type=tt, addrs=[0x0000, 0x0100], mutex_ids=[0, 1])
    output_tiles = pl.make_tile_group(
        type=tt, addrs=[0x0200, 0x0300], mutex_ids=[2, 3])

    with pl.section_vector():
        for row in pl.range(pl.get_block_idx(), x.shape[0], pl.get_block_num()):
            tile_x = input_tiles.next()
            tile_out = output_tiles.next()
            pl.load(tile_x, x, [row, 0])
            pl.add(tile_out, tile_x, tile_x)
            pl.store(out, tile_out, [row, 0])

        pl.system.sync_all(
            mode=pl.SyncAllMode.HARD,
            core_type=pl.SyncCoreType.AIV_ONLY,
        )
```

该完整示例用于展示参与AIV一致到达屏障及屏障位于运行时循环外的正确写法，并验证多核执行不会因参与者不一致而卡住；它不用于说明跨核GM数据可见性。

HARD模式不需要workspace。三种`core_type`的调用方式如下；以下片段分别用于对应类型的Kernel，不应合并为一个调用序列。示例中的实际计算应位于屏障之前或之后，不能把屏障放入运行时循环或运行时分支。

```python
# 纯Vector Kernel：所有参与AIV都执行
with pl.section_vector():
    # ... Vector阶段计算，运行时循环应在sync_all之前结束
    pl.system.sync_all(
        mode=pl.SyncAllMode.HARD,
        core_type=pl.SyncCoreType.AIV_ONLY,
    )

# 纯Cube Kernel：所有参与AIC都执行
with pl.section_cube():
    # ... Cube阶段计算，运行时循环应在sync_all之前结束
    pl.system.sync_all(
        mode=pl.SyncAllMode.HARD,
        core_type=pl.SyncCoreType.AIC_ONLY,
    )

# Cube、Vector共存的Kernel：AIC和AIV必须到达同一个MIX屏障
with pl.section_cube():
    # ... Cube阶段计算
    pl.system.sync_all(
        mode=pl.SyncAllMode.HARD,
        core_type=pl.SyncCoreType.MIX,
    )

with pl.section_vector():
    # ... Vector阶段计算
    pl.system.sync_all(
        mode=pl.SyncAllMode.HARD,
        core_type=pl.SyncCoreType.MIX,
    )
```

SOFT模式需要按`core_type`传入对应workspace，三种调用方式如下：

```python
# AIV_ONLY: GM + UB，可选在列表中加一个整数Scalar used_cores
pl.system.sync_all(
    [sync_gm, sync_ub],
    mode=pl.SyncAllMode.SOFT,
    core_type=pl.SyncCoreType.AIV_ONLY,
)

# AIC_ONLY: GM + L1
pl.system.sync_all(
    [sync_gm, sync_l1],
    mode=pl.SyncAllMode.SOFT,
    core_type=pl.SyncCoreType.AIC_ONLY,
)

# MIX: GM + UB + L1
pl.system.sync_all(
    [sync_gm, sync_ub, sync_l1],
    mode=pl.SyncAllMode.SOFT,
    core_type=pl.SyncCoreType.MIX,
)
```

SOFT `MIX`与HARD `MIX`遵循相同的参与规则：在Cube、Vector共存的Kernel中，Cube侧AIC和Vector侧AIV必须以相同顺序和相同次数调用，并传入符合上表要求的workspace。
