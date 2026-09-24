# vf.exp_sub

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

`src0`与`src1`相减，差值作为e的指数计算，根据`preg`将计算结果写入dst。公式如下：

src数据类型为DT_FP32时：

$$
dst_i = e^{(src0_i - src1_i)}
$$

src数据类型为DT_FP16时：

$$
dst_i = e^{(cast\_f16\_to\_f32(src0_i) - cast\_f16\_to\_f32(src1_i))}
$$

## 函数原型

```python
exp_sub(src0, src1, preg, layout: Optional[CastLayout] = None) -> dst
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `src0` | 输入 | 源操作数0，reg_tensor，支持的数据类型请参见[约束说明](#约束说明)。 |
| `src1` | 输入 | 源操作数1，reg_tensor，支持的数据类型请参见[约束说明](#约束说明)。 |
| `preg` | 输入 | mask_tensor。 |
| `layout` | 输入 | 可选，决定结果放置半区。`pl.CastLayout.ZERO`（偶数半区，默认）或`pl.CastLayout.ONE`（奇数半区）。<br>- src类型为DT_FP16类型时，支持`pl.CastLayout.ZERO`和`pl.CastLayout.ONE` 。<br>- src类型为DT_FP32类型时，配置不生效。|
| `dtype` | 输入 | 可选，目标reg_tensor数据类型。当src为DT_FP16时，指定`dtype=pl.DT_FP32`可将源操作数提升精度到DT_FP32再进行计算，产生DT_FP32结果。默认与源操作数类型一致。 |

## 约束说明

- 数据类型约束：

  | src0 | src1 | dst |
  |---|---|---|
  | DT_FP16 | DT_FP16 | DT_FP32 |
  | DT_FP32 | DT_FP32 | DT_FP32 |

- 源操作数数据类型为DT_FP32时，支持寄存器全部重叠；源操作数数据类型为DT_FP16时，仅支持源操作数寄存器重叠。

- 源操作数类型为DT_FP16时，Vector计算单元一次计算只处理最多64个元素，mask的有效情况以输入数据类型为准，只有偶数位有效，有效位共128bit，参考下图所示：

  **图1** exp_sub高精度示意图

  ![](../../../../figures/exp_sub_high_precision.jpg)

## 返回值说明

返回`dst`目的操作数，reg_tensor，支持的数据类型请参见[约束说明](#约束说明)。

## 调用示例

### DT_FP32源 → DT_FP32结果

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_a, src_b, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_a, 0)
    reg_b = vf.load_align(src_b, 0)
    reg_out = vf.exp_sub(reg_a, reg_b, preg)
    vf.store_align(dst_tile, reg_out, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    in_b_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    in_b = in_b_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x200, mutex_ids=[2])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_b, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf(in_a, in_b, t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    b = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, b, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.exp(a - b), rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### DT_FP16源 → DT_FP32结果

当源操作数为DT_FP16、目的操作数为DT_FP32时，寄存器中128个DT_FP16元素按相邻两两分组，`layout`决定每组中参与计算的元素位置：`pl.CastLayout.ZERO`（默认）取偶数位（第0个），`pl.CastLayout.ONE`取奇数位（第1个）。最终输出64个DT_FP32元素。以下示例使用默认的`layout=ZERO`，即取每组偶数位元素参与计算。

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_half_to_float(src_a, src_b, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    reg_a = vf.load_align(src_a, 0)
    reg_b = vf.load_align(src_b, 0)
    reg_out = vf.exp_sub(reg_a, reg_b, preg, dtype=pl.DT_FP32)
    vf.store_align(dst_tile, reg_out, preg)

@pl.jit()
def example_kernel_half_to_float(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf_in = pl.TileType(shape=[1, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tf_out = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf_in, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    in_b_grp = pl.make_tile_group(type=tf_in, addrs=0x100, mutex_ids=[1])
    in_b = in_b_grp.current()
    t_out_grp = pl.make_tile_group(type=tf_out, addrs=0x200, mutex_ids=[2])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_b, b, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf_half_to_float(in_a, in_b, t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])

def test_example_half_to_float():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 128], device=device, dtype=torch.float16)
    b = torch.randn([1, 128], device=device, dtype=torch.float16)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel_half_to_float[None, core_nums](a, b, out)
    torch.npu.synchronize()
    # layout=ZERO取偶数位（第0、2、4...个）DT_FP16元素参与计算，输出64个DT_FP32
    torch.testing.assert_close(out, torch.exp(a[:, 0::2].float() - b[:, 0::2].float()), rtol=1e-3, atol=1e-3)

if __name__ == "__main__":
    test_example_half_to_float()
    print("PASSED")
```
