# Reg矢量计算编程

Reg矢量计算直接使用SIMD Register File保存向量数据和中间结果。PyPTO Pro通过`@pl.vector_function`定义VF函数，并在函数内使用[`vf.*` API](../../../api/SIMD-API/operation/vf_computation/index.md)表达寄存器加载、计算和存储。

> [!NOTE]说明
> Reg矢量计算依赖VF Register File，使用前请确认对应VF API的支持范围。

## Reg矢量计算的适用场景

Tile向量计算以UB Tile为数据载体。多个向量操作串联时，中间结果通常需要写回UB，再由下一条指令读取。计算链较长时，反复访问UB会增加读写带宽压力和Bank冲突概率。

Reg矢量计算将一段连续计算保留在寄存器中，仅在计算链入口和出口与UB交互：

| 维度 | Tile/Membase向量计算 | Regbase向量计算 |
|:---|:---|:---|
| 数据载体 | UB中的`Tile` | Register File中的`RegTensor` / `MaskReg` |
| 中间结果 | 通常写回UB | 可由后续`vf.*`操作直接消费 |
| PyPTO Pro接口 | `pl.add`、`pl.sub`、`pl.reduce_*`等 | `vf.add`、`vf.sub`、`vf.reduce_*`等 |
| 适用场景 | 通用向量计算、快速实现 | 连续计算链、需要降低UB往返开销的高性能场景 |

## 硬件组成

Vector侧参与Reg矢量计算的硬件单元包括：

- **Reg向量执行单元**：从Register File读取操作数并将计算结果写回寄存器。
- **DMA单元**：在UB与Register File之间搬运数据。
- **Aux Scalar**：完成VF域内的地址、循环等标量计算。

**图1 SIMD Reg向量执行关系**

![SIMD Reg向量执行单元与Register File、UB的关系](../../figures/register_execution_unit.jpg)

## 内存层级

Register File位于UB之上，不能直接从GM加载或直接写回GM。完整数据路径是：

```text
GM → UB → Register File → UB → GM
```

**图2 Reg矢量计算内存层级**

![Register File、Unified Buffer和Global Memory的层级关系](../../figures/register_memory_hierarchy.jpg)

PyPTO Pro中各阶段的接口对应关系如下：

| 数据路径 | PyPTO Pro表达 |
|:---|:---|
| GM → UB | `pl.load` / `pl.load_tile` |
| UB → Register File | `vf.load` / `vf.load_align` / `vf.load_unalign`等 |
| Register File内计算 | `vf.add`、`vf.mul`、`vf.reduce_sum`等 |
| Register File → UB | `vf.store` / `vf.store_align` / `vf.store_unalign`等 |
| UB → GM | `pl.store` / `pl.store_tile` |

## 编程模型

Regbase在Tile/Membase的“数据搬入 → 计算 → 数据搬出”基础上，将向量计算阶段细分为“Load → Compute → Store”。

**图3 Regbase编程模型总体结构**

![GM、UB、Register File之间的Regbase编程流程](../../figures/regbase_programming_model_overview.jpg)

### VF函数与执行域

使用`@pl.vector_function`声明VF函数。函数体隐式处于VF执行域，只能调用`vf.*`操作；Tile参数的类型由调用点推导。VF函数通常从UB Tile加载寄存器，完成一段连续计算，再把结果存回UB Tile。

```python
import pypto_pro.language as pl


@pl.vector_function
def add_vf(src_a, src_b, dst):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_a, 0)
    reg_b = vf.load_align(src_b, 0)
    reg_out = vf.add(reg_a, reg_b, preg)
    vf.store_align(dst, reg_out, preg)
```

外层`@pl.jit` Kernel负责GM与UB之间的搬运以及跨Pipe同步，并在`pl.section_vector()`中调用VF函数：

```python
@pl.jit(auto_mutex=True)
def add_kernel(
    a: pl.Tensor[[1, 64], pl.DT_FP32],
    b: pl.Tensor[[1, 64], pl.DT_FP32],
    out: pl.Tensor[[1, 64], pl.DT_FP32],
):
    tt = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32,
                     target_memory=pl.MemorySpace.Vec)
    a_group = pl.make_tile_group(type=tt, addrs=0x000, mutex_ids=[0])
    b_group = pl.make_tile_group(type=tt, addrs=0x100, mutex_ids=[1])
    out_group = pl.make_tile_group(type=tt, addrs=0x200, mutex_ids=[2])

    with pl.section_vector():
        tile_a = a_group.current()
        tile_b = b_group.current()
        tile_out = out_group.current()
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        add_vf(tile_a, tile_b, tile_out)
        pl.store(out, tile_out, [0, 0])
```

完整可运行示例和寄存器生命周期说明参见[`vf.reg_tensor`](../../../api/SIMD-API/operation/vf_computation/reg_tensor.md)。

## 同步与依赖

- GM↔UB搬运与Vector/VF计算之间的跨Pipe依赖，由TileGroup + `auto_mutex=True`自动管理，或使用`pl.system.sync_src` / `pl.system.sync_dst`手动管理。
- VF函数内存在UB写后读、写后写等局部依赖时，按接口要求使用`vf.mem_bar`指定对应模式。
- Register File中存在直接数据依赖的`vf.*`表达式应保持清晰的数据流关系，避免在未初始化寄存器上执行计算。

## 使用建议

满足以下条件时优先考虑Reg矢量计算：

1. 目标设备和所需VF API均受支持。
2. 性能热点由较长的连续向量计算链构成。
3. Tile/Membase实现中存在明显的中间结果UB往返。
4. 经过性能分析确认寄存器方案能带来收益。

普通向量计算仍建议先使用[Tile矢量计算](Tile_vector_computation.md)完成正确实现，再针对热点替换为VF计算。
