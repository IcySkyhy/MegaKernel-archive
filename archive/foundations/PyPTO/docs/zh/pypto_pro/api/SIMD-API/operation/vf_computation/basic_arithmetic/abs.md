# vf.abs

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

- 对实数类型（DT_INT8、DT_INT16、DT_INT32、DT_FP16、DT_FP32）

    对`src`中的有效元素逐个取绝对值，并将结果写入`dst`对应位置，计算公式如下：

    $$dst_i = |src_i|$$

- 对复数类型（DT_FP16双寄存器模式、DT_FP32双寄存器模式）

    复数类型通过双寄存器模式实现：DT_FP16双寄存器模式对应由两个DT_FP16组成的复数（实部和虚部各16位，共32位），DT_FP32双寄存器模式对应由两个DT_FP32组成的复数（实部和虚部各32位，共64位）。`reg[0]`存储实部，`reg[1]`存储虚部。

    对`src`中有效元素逐个取模，并将结果写入`dst`对应位置，计算公式如下：

    $$dst_i = |src_i| = (\alpha^2 + \beta^2)^{1/2}$$

    其中$src_i = \alpha + \beta i$，α为复数的实部，β为复数的虚部。

## 函数原型

```python
abs(src, preg, mode: Optional[MergeMode] = None) -> dst
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `src` | 输入 | 源操作数，reg_tensor。源操作数`src`与目的操作数`dst`的数据类型保持一致。支持的数据类型为：DT_INT8、DT_INT16、DT_INT32、DT_FP16、DT_FP32。其中DT_FP16和DT_FP32支持双寄存器模式，用于复数取模运算。 |
| `preg` | 输入 | mask_tensor。 |
| `mode` | 输入 | 可选，对应[MergeMode](../types/MergeMode.md)类型。<br>- `pl.MergeMode.ZEROING`（默认），`preg`未筛选的元素在`dst`中置0。<br>- `pl.MergeMode.MERGING`当前不支持。 |

## 约束说明

- 当目的操作数和源操作数数据类型不一致时，目的操作数和源操作数不可重叠。

## 返回值说明

返回`dst`目的操作数，reg_tensor，支持的数据类型和`src`中的说明一致。整型数据的计算结果如果超出数据类型的表示范围会采取非饱和截断，比如DT_INT8类型，src为-128，其绝对值128会被截断成-128。

## 调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_tile, 0)
    reg_out = vf.abs(reg_a, preg)
    vf.store_align(dst_tile, reg_out, preg)

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
    torch.testing.assert_close(out, torch.abs(a), rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
