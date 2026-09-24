# vf.get_mask_spr

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

从`pl.set_vec_mask`设置的掩码寄存器 {MASK1, MASK0} 中读取mask值，并按数据类型对应的格式转换后写入返回值`mask_reg`。

本接口对应AscendC `MoveMask<T>`接口。具体转换方式：

- **32位宽（DT_INT32、DT_UINT32、DT_FP32）**：读取64bit的MASK0数据，将每个bit复制为4bit，写入`mask_reg`。
- **16位宽（DT_INT16、DT_UINT16、DT_FP16、DT_BF16）**：读取完整128bit的 {MASK1, MASK0} 数据，将每个bit复制为2bit，写入`mask_reg`。

## 函数原型

```python
get_mask_spr(width: MaskWidth = MaskWidth.B32) -> mask_reg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `width` | 输入 | 掩码宽度，决定读取的SPR位宽及扩展方式，对应[MaskWidth](../types/MaskWidth.md)类型。<br>- `pl.MaskWidth.B32`（默认）：读取64bit MASK0，每bit扩展为4bit，对应`movp_b32()`指令。<br>- `pl.MaskWidth.B16`：读取128bit {MASK1, MASK0}，每bit扩展为2bit，对应`movp_b16()`指令。 |

## 约束说明

- 数据类型约束：

  | width | 返回值 |
  |---|---|
  | pl.MaskWidth.B32 | mask_tensor（对应32位宽：DT_INT32、DT_UINT32、DT_FP32） |
  | pl.MaskWidth.B16 | mask_tensor（对应16位宽：DT_INT16、DT_UINT16、DT_FP16、DT_BF16） |

- 本接口为兼容性接口，建议优先采用`vf.create_mask`和`vf.update_mask`进行mask_tensor计算。

- 本接口使用前需选择与掩码含义一致的模式，并通过`pl.set_vec_mask`或产生掩码的VF指令设置SPR {MASK1, MASK0}。按位掩码使用norm模式；元素计数使用count模式。

## 返回值说明

返回`mask_reg`返回的mask_tensor，从SPR {MASK1, MASK0} 读取并转换。

## 调用示例

先用`pl.set_vec_mask`设置SPR {MASK1, MASK0}，再用`vf.get_mask_spr`读取到mask_tensor，最后用该mask_tensor控制计算：

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg = vf.load_align(src_tile, 0)
    # 从SPR读取掩码到mask_tensor（movp_b32指令）
    spr_mask = vf.get_mask_spr(width=pl.MaskWidth.B32)
    # 使用读取的掩码做abs：前32个元素取abs，其余置零
    reg_dst = vf.abs(reg, spr_mask)
    vf.store_align(dst_tile, reg_dst, preg)

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
        pl.set_mask_norm()
        pl.set_vec_mask(0, 0xFFFFFFFF)
        example_vf(in_a, t_out)
        pl.reset_mask()
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
    # 前32个元素取abs，后32个置零
    expected = torch.zeros_like(a)
    expected[:, :32] = torch.abs(a[:, :32])
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
