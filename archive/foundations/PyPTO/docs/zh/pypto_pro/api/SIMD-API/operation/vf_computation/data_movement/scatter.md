# vf.scatter

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

该指令会根据索引值`index`将源操作数`src`中的元素分散到目的操作数`tile`中。分散过程如下图所示：

**图1** vf.scatter功能说明

![vf.scatter功能说明](../../../../figures/scatter_function.jpg)

## 函数原型

```python
scatter(tile, src, index, preg)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `tile` | 输出 | 目的操作数，Tile地址，需要32字节对齐。 |
| `src` | 输入 | 源操作数，reg_tensor，支持的数据类型请参见[约束说明](#约束说明)。当src为DT_INT8或者DT_UINT8数据类型时，源操作数中仅偶数位元素有效。即src中的偶数位置[0, 2, 4, ..., 252, 254]的数据会被分散存储到目的操作数中。 |
| `index` | 输入 | 索引值，reg_tensor，支持的数据类型请参见[约束说明](#约束说明)，src中的每个元素在Tile中相对于基地址的位置，单位：元素个数。`index`中的值必须唯一，若存在重复的`index`值，系统仅保留其中一个对应的数据。 |
| `preg` | 输入 | mask_tensor。 |

## 约束说明

- 数据类型约束：

  | src | index |
  |---|---|
  | DT_INT8 | DT_UINT16 |
  | DT_UINT8 | DT_UINT16 |
  | DT_INT16 | DT_UINT16 |
  | DT_UINT16 | DT_UINT16 |
  | DT_FP16 | DT_UINT16 |
  | DT_BF16 | DT_UINT16 |
  | DT_INT32 | DT_UINT32 |
  | DT_UINT32 | DT_UINT32 |
  | DT_FP32 | DT_UINT32 |
  | DT_INT64 | DT_UINT32 |
  | DT_INT64 | DT_UINT64 |
  | DT_UINT64 | DT_UINT32 |
  | DT_UINT64 | DT_UINT64 |

## 返回值说明

无

## 调用示例

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu

@pl.vector_function
def example_vf(src_tile, index_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    src_reg = vf.load_align(src_tile, 0)
    index_reg = vf.load_align(index_tile, 0)
    # 根据索引将src_reg中的元素分散存储到dst_tile
    vf.scatter(dst_tile, src_reg, index_reg, preg)

@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    idx: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_UINT32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tf_idx = pl.TileType(shape=[1, 64], dtype=pl.DT_UINT32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    in_idx_grp = pl.make_tile_group(type=tf_idx, addrs=0x100, mutex_ids=[1])
    in_idx = in_idx_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x200, mutex_ids=[2])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.load(in_idx, idx, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf(in_a, in_idx, t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])

def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    idx = torch.arange(64, device=device, dtype=torch.int32).reshape([1, 64])
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, idx, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
