# vf.arange

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

从起始值`start`生成索引序列，用于构造索引向量。通过`index_order`选择生成方向：

- `pl.IndexOrder.INCREASE_ORDER`（默认）：递增，`dst[i] = start + i`。
- `pl.IndexOrder.DECREASE_ORDER`：递减，`dst[i] = start - i`。

## 函数原型

```python
arange(start, index_order: Optional[IndexOrder] = None, dtype: Optional[DType] = None) -> dst
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `start` | 输入 | 序列起始值（整型标量或表达式）。 |
| `index_order` | 输入 | 可选关键字参数，生成方向。`pl.IndexOrder.INCREASE_ORDER`（默认，递增）或`pl.IndexOrder.DECREASE_ORDER`（递减）。每一步的步长固定为 ±1，这是硬件特性。如需非1步长（如`start + i*step`），可在`vf.arange`之后追加一条`vf.muls`对结果整体缩放。 |
| `dtype` | 输入 | 必选，指定目标reg_tensor的数据类型（如`pl.DT_UINT32`、`pl.DT_INT32`等）。由于标量源无法推断寄存器数据类型，必须显式指定。 |

## 约束说明

- 数据类型约束：

  支持的dst数据类型：DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_INT32、DT_UINT32、DT_FP16、DT_FP32。

## 返回值说明

返回`dst`目标reg_tensor，支持的数据类型请参见[约束说明](#约束说明)，存放生成的序列。

## 调用示例

### 递增序列

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    idxC = vf.arange(0, dtype=pl.DT_UINT32)
    vf.store_align(dst_tile, idxC, preg)

@pl.jit()
def example_kernel(
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
):
    tu = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    t_out_grp = pl.make_tile_group(type=tu, addrs=0x0, mutex_ids=[0])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf(t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    out = torch.empty([1, 64], device=device, dtype=torch.int32)
    example_kernel[None, core_nums](out)
    torch.npu.synchronize()
    expected = torch.arange(64, dtype=torch.int32).unsqueeze(0).to(device)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### 递减序列

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf_dec(dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    idxC = vf.arange(63, index_order=pl.IndexOrder.DECREASE_ORDER, dtype=pl.DT_UINT32)
    vf.store_align(dst_tile, idxC, preg)

@pl.jit()
def example_kernel_dec(
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
):
    tu = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    t_out_grp = pl.make_tile_group(type=tu, addrs=0x0, mutex_ids=[0])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf_dec(t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])

def test_example_2():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    out = torch.empty([1, 64], device=device, dtype=torch.int32)
    example_kernel_dec[None, core_nums](out)
    torch.npu.synchronize()
    expected = torch.arange(126, 62, -1, dtype=torch.int32).unsqueeze(0).to(device)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

if __name__ == "__main__":
    test_example_2()
    print("PASSED")
```
