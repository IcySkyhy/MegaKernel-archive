# vf.full

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

将标量值或源reg_tensor的`src`最低/最高位元素广播到目标reg_tensor的`dst`各个元素。支持两种模式：

- **Scalar模式**：将标量值广播到寄存器各元素。
- **Tensor模式**：将源reg_tensor的最低位或最高位元素广播到目标reg_tensor各元素。Tensor模式必须带掩码。

## 函数原型

```python
full(src, preg=None, dtype: Optional[DType] = None, mode: Optional[MergeMode] = None, pos: Optional[DuplicatePos] = None) -> dst
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `src` | 输入 | 源操作数，为标量值或者reg_tensor。源操作数`src`与目的操作数`dst`的数据类型保持一致。<br>- **Scalar模式**：标量值，广播到寄存器各元素。支持的数据类型为：DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_FP16、DT_BF16、DT_INT32、DT_UINT32、DT_FP32。<br>- **Tensor模式**：reg_tensor，广播其最低位或最高位元素。支持的数据类型为：DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_FP16、DT_BF16、DT_INT32、DT_UINT32、DT_FP32、DT_FP8E4M3FN、DT_FP8E5M2、DT_FP8E8M0、DT_HF8、DT_FP4E2M1、DT_FP4E1M2。 |
| `preg` | 输入 | mask_tensor。Tensor模式必选；Scalar模式可选。 |
| `dtype` | 输入 | 数据类型。<br>- Scalar模式必选，指定目标reg_tensor的数据类型。<br>- Tensor模式由源reg_tensor自动推断，无需指定。 |
| `pos` | 输入 | 可选，Tensor模式下选择广播源reg_tensor的哪个元素，对应[DuplicatePos](../types/DuplicatePos.md)类型：<br>- `pl.DuplicatePos.LOWEST`：默认，广播最低位的元素。<br>- `pl.DuplicatePos.HIGHEST`：指定广播最高位的元素。 |
| `mode` | 输入 | 可选，对应[MergeMode](../types/MergeMode.md)类型。<br>- `pl.MergeMode.ZEROING`（默认），`preg`未筛选的元素在`dst`中置0。<br>- `pl.MergeMode.MERGING`，`preg`未筛选的元素在`dst`中保留原值。 |

## 约束说明

无

## 返回值说明

返回`dst`目的操作数，reg_tensor，支持的数据类型和`src`中的说明一致。

## 调用示例

### Scalar模式

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    # 标量广播到reg_tensor各元素，可不带掩码或带掩码
    max0 = vf.full(3.0, dtype=pl.DT_FP32)
    sum0 = vf.full(0.0, preg, dtype=pl.DT_FP32)
    vf.store_align(dst_tile, max0, preg)

@pl.jit()
def example_kernel(
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    t_out = t_out_grp.current()
    with pl.section_vector():
        example_vf(t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, torch.full([1, 64], 3.0, device=device, dtype=torch.float32), rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### Tensor模式 — FP8E4M3FN 寄存器广播

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_fp8(src_tile, dst_tile):
    # full 使用 b8 掩码（FP8 元素宽度）
    preg_f8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP8E4M3FN)
    # astype/store 使用 b32 掩码（FP32 元素宽度）
    preg_f32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    # 加载 FP8E4M3FN 数据，reg_tensor 包含 256 个 FP8 元素
    reg_f8 = vf.load_align(src_tile, 0, dtype=pl.DT_FP8E4M3FN)
    # Tensor 模式：广播最低位 FP8 元素到所有 256 个 lane
    reg_dup = vf.full(reg_f8, preg_f8)
    # FP8 → FP32 转换（4x 扩展，256 个 FP8 → 64 个 FP32）
    reg_f32 = vf.astype(reg_dup, preg_f32, dtype=pl.DT_FP32)
    vf.store_align(dst_tile, reg_f32, preg_f32)

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
    # full 广播最低位 FP8 元素到所有 lane，astype layout=ZERO 取每 4 个 FP8 中的第 0 个
    # 所有输出元素均等于第一个 FP8 元素的 FP32 值
    expected = a[:, :1].to(torch.float32).expand([1, 64])
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

if __name__ == "__main__":
    test_example_fp8()
    print("PASSED")
```

### Tensor模式 — FP4E1M2 寄存器广播

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_fp4(src_tile, dst_tile):
    # full 使用 b8 掩码（FP4 以 b8 打包存储）
    preg_f4 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP4E1M2)
    # astype/store 使用 b16 掩码（BF16 元素宽度）
    preg_bf16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    # 以 FP4E1M2 类型加载，reg_tensor 包含 256 个 b8 元素（512 个 FP4）
    reg_f4 = vf.load_align(src_tile, 0, dtype=pl.DT_FP4E1M2)
    # Tensor 模式：广播最低位 b8 元素到所有 256 个 lane
    reg_dup = vf.full(reg_f4, preg_f4)
    # FP4 → BF16 转换（2x 扩展，256 个 b8 → 128 个 BF16）
    reg_bf16 = vf.astype(reg_dup, preg_bf16, dtype=pl.DT_BF16)
    vf.store_align(dst_tile, reg_bf16, preg_bf16)

@pl.jit()
def example_kernel_fp4(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT8],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
):
    tf_in = pl.TileType(shape=[1, 256], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec)
    tf_out = pl.TileType(shape=[1, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf_in, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf_out, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf_fp4(in_a, t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])

def test_example_fp4():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    # FP4E1M2 以 b8 打包存储（2 个 FP4/字节），0x44 编码两个 1.0 值（code=4）
    a = torch.randint(0, 256, [1, 256], device=device, dtype=torch.uint8)
    a[:, 0] = 0x44  # 首字节编码 FP4 1.0
    out = torch.empty([1, 128], device=device, dtype=torch.bfloat16)
    example_kernel_fp4[None, core_nums](a, out)
    torch.npu.synchronize()
    # full 广播首字节 0x44 到所有 lane，所有 FP4 均为 1.0，转换为 BF16 后全为 1.0
    expected = torch.ones([1, 128], device=device, dtype=torch.bfloat16)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

if __name__ == "__main__":
    test_example_fp4()
    print("PASSED")
```

### Tensor模式 — HF8 寄存器广播

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_hf8(src_tile, dst_tile):
    # full 使用 b8 掩码（HF8 元素宽度）
    preg_hf8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_HF8)
    # astype/store 使用 b16 掩码（FP16 元素宽度）
    preg_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    # 加载 HF8 数据，reg_tensor 包含 256 个 HF8 元素
    reg_hf8 = vf.load_align(src_tile, 0, dtype=pl.DT_HF8)
    # Tensor 模式：广播最低位 HF8 元素到所有 256 个 lane
    reg_dup = vf.full(reg_hf8, preg_hf8)
    # HF8 → FP16 转换（2x 扩展，256 个 HF8 → 128 个 FP16）
    reg_f16 = vf.astype(reg_dup, preg_f16, dtype=pl.DT_FP16)
    vf.store_align(dst_tile, reg_f16, preg_f16)

@pl.jit()
def example_kernel_hf8(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_HF8],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    tf_in = pl.TileType(shape=[1, 256], dtype=pl.DT_HF8, target_memory=pl.MemorySpace.Vec)
    tf_out = pl.TileType(shape=[1, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf_in, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf_out, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf_hf8(in_a, t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])

def test_example_hf8():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    # 创建 HF8 数据（1.0 在 HF8 中可精确表示）
    a = torch.ones([1, 256], device=device, dtype=torch.float32)
    a = torch_npu.npu_dtype_cast(a, torch_npu.hifloat8)
    out = torch.empty([1, 128], device=device, dtype=torch.float16)
    example_kernel_hf8[None, core_nums](a, out)
    torch.npu.synchronize()
    # full 广播最低位 HF8 元素到所有 lane，转换为 FP16 后全为 1.0
    expected = torch.ones([1, 128], device=device, dtype=torch.float16)
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)

if __name__ == "__main__":
    test_example_hf8()
    print("PASSED")
```

### Tensor模式 — FP8E8M0 寄存器广播

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_fp8e8m0(src_tile, dst_tile):
    # full/store 使用 b8 掩码（FP8E8M0 元素宽度）
    preg_f8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP8E8M0)
    # 以 FP8E8M0 类型加载，reg_tensor 包含 256 个 FP8E8M0 元素
    reg_f8 = vf.load_align(src_tile, 0, dtype=pl.DT_FP8E8M0)
    # Tensor 模式：广播最低位 FP8E8M0 元素到所有 256 个 lane
    reg_dup = vf.full(reg_f8, preg_f8)
    # 直接存储 FP8E8M0（b8 宽度，无 astype 转换）
    vf.store_align(dst_tile, reg_dup, preg_f8)

@pl.jit()
def example_kernel_fp8e8m0(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT8],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT8],
):
    tf_in = pl.TileType(shape=[1, 256], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec)
    tf_out = pl.TileType(shape=[1, 256], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf_in, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf_out, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf_fp8e8m0(in_a, t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])

def test_example_fp8e8m0():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    # FP8E8M0 以 b8 存储。首字节设为 0x7E（exponent=127，对应 scale=2^0=1.0）
    a = torch.zeros([1, 256], device=device, dtype=torch.uint8)
    a[:, 0] = 0x7E
    out = torch.empty([1, 256], device=device, dtype=torch.uint8)
    example_kernel_fp8e8m0[None, core_nums](a, out)
    torch.npu.synchronize()
    # full 广播首字节 0x7E 到所有 256 个 lane
    expected = torch.full([1, 256], 0x7E, device=device, dtype=torch.uint8)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

if __name__ == "__main__":
    test_example_fp8e8m0()
    print("PASSED")
```
