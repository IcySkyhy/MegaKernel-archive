# vf.load_unalign

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

为提升对不规则内存地址的处理能力，reg_tensor支持在数据搬运过程中对非32字节对齐的地址进行访问，降低非对齐访问带来的性能开销。`vf.load_unalign`能够实现数据从非对齐的地址连续搬运至reg_tensor，利用非对齐寄存器UnalignRegForLoad作为临时缓存区，暂存跨对齐边界的数据，从而实现高效的连续非对齐数据传输。

非对齐搬入有三类接口：普通搬运接口、PostUpdate扩展搬运接口、使用AddrReg寄存器存储偏移量接口。

| 接口类型 | 触发条件 | 说明 |
|---|---|---|
| 普通搬运接口 | 不传`stride`，`post_update=False`（默认） | 完成一次搬运后，Tile地址不会自动更新，每次迭代需要手动更新地址。 |
| PostUpdate扩展搬运接口 | `post_update=True`或传入`stride` | 完成一次搬运后，Tile地址会自动更新，每次迭代不需要手动更新地址。 |
| AddrReg存储偏移量接口 | offset为`vf.create_addr_reg`创建的AddrReg | 在每次迭代中，需要先调用`vf.create_addr_reg`手动设定地址偏移量，再调用搬运指令。 |

在读非对齐地址前，应该先通过`vf.load_unalign_pre`进行初始化，保存非32字节对齐的数据，然后再调用`vf.load_unalign`进行数据搬入。

### 非对齐搬入原理

如下图所示，从Tile地址srcAddr ~ 304读取数据，并将其搬运至目标reg_tensor（256B）。处理流程如下：

① 调用**load_unalign_pre**进行非对齐搬入初始化。非对齐寄存器ureg缓存Tile地址32 ~ 64的有效数据，作为后续非对齐访问的前置数据缓存。

② 调用**load_unalign**，硬件指令将Tile地址64 ~ 320的对齐数据搬入临时reg_tensor，并将ureg中srcAddr ~ 64对应的数据与临时reg_tensor中地址64 ~ 304对应的数据拼接在一起，将结果写入目标reg_tensor。此外，Tile地址288 ~ 320的数据会被写入ureg。

**图1** 非对齐搬入示例

![](../../../../figures/unaligned_load.jpg)

### 连续非对齐搬入搬出原理

**图2** 连续非对齐搬入搬出原理（数据类型DT_UINT32）

![](../../../../figures/contiguous_unaligned_load_store.jpg)

连续非对齐搬入时，`vf.load_unalign`会将后续未对齐的数据缓存至`vf.load_unalign_init`创建的ureg，所以下一次搬入不需要再次调用`vf.load_unalign_pre`，只需在迭代开始前调用一次`vf.load_unalign_pre`，从而实现非对齐搬入的性能优化。

连续非对齐搬出时，下次迭代的`vf.store_unalign`会将本次迭代`vf.store_unalign`缓存至ureg中的数据写入Tile，所以本次迭代不需要调用`vf.store_unalign_post`将ureg数据写入Tile，只需在迭代结束后调用一次`vf.store_unalign_post`，从而实现非对齐搬出的性能优化。

如上图所示，将Tile地址48 ~ 560的DT_UINT32数据[1, 2, 3, ..., 128]搬入至dstReg，再搬回Tile，需要两次搬入搬出操作，即for循环执行两次，初始化和后处理移至for循环外。stride = 256B / sizeof(T)（即每次地址偏移256B），repeatTimes = dataSize / 256B（即迭代次数=总数据量/VL）。

具体的搬运步骤如下：

1. 非对齐搬入初始化：更新ureg = [1, 2, 3, 4]；
2. 非对齐搬入：tmpReg = [5, 6, 7, ..., 68]，tmpReg部分数据和ureg数据写入dstReg = [1, 2, 3, ..., 64]，更新ureg = [61, 62, 63, ..., 68]；
3. 非对齐搬出：dstReg部分数据[1, 2, 3, ..., 60]写入Tile地址48 ~ 288，更新align_reg = [61, 62, 63, 64]；
4. 非对齐搬入：tmpReg = [69, 70, 71, ..., 128]，tmpReg数据和ureg部分数据写dstReg = [65, 66, 67, ..., 128]；
5. 非对齐搬出：align_reg数据[61, 62, 63, 64]和dstReg部分数据[65, 66, 67, ..., 124]写入Tile地址288 ~ 544，更新align_reg = [125, 126, 127, 128]；
6. 非对齐搬出后处理：将align_reg中缓存的数据[125, 126, 127, 128]写入Tile地址544 ~ 560。

## 函数原型

```python
load_unalign(tile, align_reg, stride=None, post_update: bool = False) -> dst
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `tile` | 输入 | 源操作数，Tile地址，起始地址不需要32字节对齐。目的操作数与源操作数的数据类型需要保持一致。支持的数据类型为：DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_FP16、DT_BF16、DT_INT32、DT_UINT32、DT_FP32、DT_INT64、DT_UINT64、DT_FP8E4M3FN、DT_FP8E5M2、DT_FP8E8M0、DT_HF8、DT_FP4E2M1、DT_FP4E1M2。 |
| `align_reg` | 输入/输出 | 非对齐寄存器，UnalignRegForLoad类型，用于存储非32字节的数据，寄存器大小为32字节（由`vf.load_unalign_init()`创建）。 |
| `stride` | 输入 | 可选，地址更新步长，单位：字节。仅在`post_update=True`时有效。 |
| `post_update` | 输入 | 可选，`True`时搬运后地址自动累进，默认`False`。 |

## 约束说明

- `vf.load_unalign_pre`与`vf.load_unalign`接口需要组合使用。

## 返回值说明

返回`dst`目的操作数，reg_tensor，支持的数据类型请参见[约束说明](#约束说明)。

## 调用示例

### 基本非对齐加载

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    ureg = vf.load_unalign_init()
    vf.load_unalign_pre(ureg, src_tile)
    # 非对齐搬入：ureg缓存跨对齐边界数据
    src_reg = vf.load_unalign(ureg, src_tile, post_update=True)
    store_ureg = vf.unalign_reg_for_store()
    vf.store_unalign(dst_tile, src_reg, store_ureg, 64, post_update=True)
    vf.store_unalign_post(dst_tile, store_ureg, 0, post_update=True)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf(in_a, t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### 带步长的连续非对齐加载示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    # 非对齐搬入初始化，只需在迭代开始前调用一次
    ureg = vf.load_unalign_init()
    vf.load_unalign_pre(ureg, src_tile)
    # 非对齐搬出初始化
    store_ureg = vf.unalign_reg_for_store()
    # 带步长形式：stride=64指定每次搬运后地址累进64个元素
    src_reg = vf.load_unalign(ureg, src_tile, 64, post_update=True)
    # 非对齐搬出配套使用
    vf.store_unalign(dst_tile, src_reg, store_ureg, 64, post_update=True)
    # 循环结束后刷出剩余数据
    vf.store_unalign_post(dst_tile, store_ureg, 64, post_update=True)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf(in_a, t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])

def test_example_2():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example_2()
    print("PASSED")
```
