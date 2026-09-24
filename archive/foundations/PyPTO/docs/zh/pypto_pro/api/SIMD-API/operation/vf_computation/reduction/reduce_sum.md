# vf.reduce_sum

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

reg_tensor求和归约：将源寄存器`src`中的所有有效元素（`preg`选中的元素）求和，结果写入目标寄存器的第一个元素`dst[0]`，其余元素置零。

以二叉树累加的方式计算源操作数`src`内有效元素的数据总和。以DT_FP16类型的数据求和为例，在`src`内有128个数，通过二叉树的方式，两两相加，最终得到目的操作数为1个DT_FP16类型的数据sum，计算过程如下图所示：

**图1** reduce_sum累加顺序

![reduce_sum累加顺序](../../../../figures/reduce_sum_accum_order.jpg)

当源操作数数据类型为DT_FP16时，中间累加过程在DT_FP32精度下进行，最终结果再舍入为DT_FP16，因此精度高于先逐对`vf.add`再手动归约的写法。

## 函数原型

```python
reduce_sum(src, preg, datablock: bool = False, merge_mode: Optional[MergeMode] = None)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `src` | 输入 | 源reg_tensor，源操作数`src`与目的操作数`dst`的数据类型保持一致。支持的数据类型为：DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_FP16、DT_FP32。 |
| `preg` | 输入 | mask_tensor。当所有元素均不参与计算时（mask为空），将目的操作数数据类型的0写入`dst[0]`。 |
| `datablock` | 输入 | 可选，决定接口工作模式，`True`时按datablock粒度归约（对应`vcgadd`指令），默认`False`。当`datablock=True`时，启用datablock粒度归约，每个datablock独立归约：32位宽（DT_INT32、DT_UINT32、DT_FP32）类型每16个元素为一个datablock，16位宽（DT_INT16、DT_UINT16、DT_FP16）类型每32个元素为一个datablock，各datablock分别求和并将结果写入各自datablock的第一个元素。 |
| `merge_mode` | 输入 | 可选，对应[MergeMode](../types/MergeMode.md)类型。<br>- `pl.MergeMode.ZEROING`（默认），`preg`未筛选的元素在`dst`中置0。<br>- `pl.MergeMode.MERGING`当前不支持。 |

## 约束说明

- `datablock=True`时，支持的数据类型为：DT_INT16、DT_UINT16、DT_FP16、DT_INT32、DT_UINT32、DT_FP32。

## 返回值说明

返回`dst`目标reg_tensor，支持的数据类型和`src`中的说明一致，归约结果写入第一个元素`dst[0]`，其余元素置零。指令内累加顺序采用二叉树累加方式，结果具有确定性。

## 调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    src0 = vf.load_align(src_tile, 0)
    sum0 = vf.reduce_sum(src0, preg_all)
    vf.store_align(dst_tile, sum0, preg_all)

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
    torch.testing.assert_close(out[0, 0], a.sum(), rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
