# vf.move

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

将源操作数`src`中的元素复制到目标操作数`dst`的对应位置。支持reg_tensor和mask_tensor两种寄存器类型：

- **reg_tensor模式**：对src中的有效元素逐个复制写入dst中对应位置，无效位置保留dst原值。
- **mask_tensor模式**：将src中的bit复制到dst中对应位置。如果有输入mask，则仅复制被mask选定的有效bit，无效位置填0。机制如下图所示：16位宽（DT_INT16、DT_UINT16、DT_FP16、DT_BF16）类型读取完整128bit的 {MASK1, MASK0}，将每个bit复制为2bit；32位宽（DT_INT32、DT_UINT32、DT_FP32）类型读取64bit的MASK0，并将每个bit复制为4bit。

  ![](../../../../figures/move_mask_mode.jpg)

## 函数原型

```python
move(src, preg=None, mode: Optional[MergeMode] = None) -> dst
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `src` | 输入 | 源操作数，reg_tensor或mask_tensor类型，源操作数`src`与目的操作数`dst`的数据类型保持一致。支持的数据类型为：DT_BOOL、DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_FP16、DT_BF16、DT_INT32、DT_UINT32、DT_FP32。 |
| `preg` | 输入 | 可选，mask_tensor。控制哪些元素/bit参与操作：<br>- **reg_tensor模式**：`preg`为元素操作的有效指示。`preg`选中的位置，将`src`中对应元素复制写入`dst`；`preg`未选中的位置，`dst`保留原值。<br>- **mask_tensor模式**：`preg`控制哪些bit有效。`preg`选中的bit，将`src`中对应bit复制到`dst`；`preg`未选中的bit，`dst`对应位置填0。 |
| `mode` | 输入 | 可选，对应[MergeMode](../types/MergeMode.md)类型。<br>- `pl.MergeMode.MERGING`（默认），`preg`未选中的元素在`dst`中保留原值。<br>- `pl.MergeMode.ZEROING`，当前**不支持**。 |

## 约束说明

无

## 返回值说明

返回`dst`目的操作数，reg_tensor或mask_tensor类型，支持的数据类型请参见[约束说明](#约束说明)。

## 调用示例

### reg_tensor模式

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    # reg_tensor move — 将reg_a的内容复制到reg_b
    reg_a = vf.load_align(src_tile, 0)
    reg_b = vf.move(reg_a, preg)
    src_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    dst_mask = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_FP32)
    dst_mask = vf.move(src_mask)
    vf.store_align(dst_tile, reg_b, dst_mask)

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

### mask_tensor模式

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg = vf.load_align(src_tile, 0)
    # 生成比较掩码：reg >= 0的位置为1
    mask_a = vf.ge(reg, 0.0, preg)
    # 复制掩码
    dst_mask = vf.move(mask_a)
    # 使用复制后的掩码做abs：reg >= 0处取abs（即自身），否则置零
    reg_dst = vf.abs(reg, dst_mask)
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
    expected = torch.where(a >= 0, a, torch.zeros_like(a))
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### reg_tensor模式 FP8 数据类型

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_fp8(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    # 加载 FP8E4M3FN 数据，reg_tensor 包含 256 个 FP8 元素
    reg_f8 = vf.load_align(src_tile, 0, dtype=pl.DT_FP8E4M3FN)
    # FP8 → FP32 转换（4x 扩展，256 个 FP8 → 64 个 FP32）
    reg_f32 = vf.astype(reg_f8, preg, dtype=pl.DT_FP32)
    # move 搬运 FP32 数据
    reg_dst = vf.move(reg_f32, preg)
    vf.store_align(dst_tile, reg_dst, preg)

@pl.jit()
def example_kernel_fp8(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP8E4M3FN],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf_in = pl.TileType(shape=[1, 256], dtype=pl.DT_FP8E4M3FN, target_memory=pl.MemorySpace.Vec)
    tf_out = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf_in, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf_out, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf_fp8(in_a, t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])

def test_example_fp8():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 256], device=device, dtype=torch.float32).to(torch.float8_e4m3fn)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel_fp8[None, core_nums](a, out)
    torch.npu.synchronize()
    expected = a.to(torch.float32)
    # FP8→FP32 4x 扩展，layout=ZERO(PART_P0) 取 FP8 索引 0,4,8,...,252
    torch.testing.assert_close(out, expected[:, ::4], rtol=1e-2, atol=1e-2)

if __name__ == "__main__":
    test_example_fp8()
    print("PASSED")
```
