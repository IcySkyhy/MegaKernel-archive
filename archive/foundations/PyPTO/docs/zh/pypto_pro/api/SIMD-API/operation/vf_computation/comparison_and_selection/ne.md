# vf.ne

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

逐元素比较两个操作数`a`、`b`是否不相等，将比较结果写入目的操作数`dst_mask`中对应比特位，如果比较后的结果为真，则输出结果的对应比特位为1，否则为0。

第二个参数可以是标量或reg_tensor，接口自动识别并分发到对应的硬件指令。

$$dstReg_i = \begin{cases} 1 & \text{if } a_i \neq b_i \\ 0 & \text{otherwise} \end{cases}$$

## 函数原型

```python
ne(a, b, preg, cmp_dtype: Optional[DType] = None) -> dst_mask
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `a` | 输入 | 源操作数，reg_tensor。支持的数据类型为：DT_INT8、DT_UINT8、DT_INT16、DT_UINT16、DT_FP16、DT_BF16、DT_INT32、DT_UINT32、DT_FP32。`a`和`b`可以是同一个reg_tensor。 |
| `b` | 输入 | 比较操作数，可以是标量或reg_tensor，数据类型与`a`一致。 |
| `preg` | 输入 | mask_tensor，指定参与比较的元素范围。通过`preg`参数控制的未选中元素在目的操作数中被置零。 |
| `cmp_dtype` | 输入 | 可选关键字参数，向量比较时指定比较位宽的数据类型。若未传入，则根据`a`的dtype自动推断；若传入，则按指定数据类型宽度进行比较。例如将DT_UINT16寄存器按DT_UINT8宽度比较时，传入`cmp_dtype=pl.DT_UINT8`。 |

## 约束说明

无

## 返回值说明

返回`dst_mask`目标mask_tensor，存放比较结果。

## 调用示例

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
    # 向量比较
    dst_mask = vf.ne(reg_a, reg_b, preg)
    # 标量比较：dst_mask = vf.ne(reg_a, 0.0, preg)
    reg_out = vf.select(reg_a, reg_b, dst_mask)
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
    torch.testing.assert_close(out, torch.where(a != b, a, b), rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```
