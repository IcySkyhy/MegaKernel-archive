# vf.reg_tensor

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

向量寄存器，VF计算的基本数据容器。用于存储从UB tile加载的数据、VF运算的中间结果和最终输出。

`vf.reg_tensor`是`@pl.vector_function`装饰的函数内的保留命名空间`vf`提供的核心类型，无需额外import。

## 函数原型

`vf.reg_tensor`为类型声明，不能直接调用，由编译器在赋值形式中自动声明。通过以下接口的赋值形式创建：

- 从UB tile加载数据

```python
# 不指定 dtype（由 tile 数据类型推断）
reg = vf.load_align(tile, offset)

# 指定 dtype
reg = vf.load_align(tile, offset, dtype=pl.DT_UINT32)
```

- 标量初始化

```python
reg = vf.full(scalar, preg, dtype)
```

- VF运算结果赋值

```python
reg = vf.add(src0, src1, preg)
reg = vf.mul(src0, src1, preg)
# ...其他 VF 计算接口
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dtype` | 输入 | 寄存器存储的数据类型，决定寄存器覆盖的元素个数。 - `vf.reg_tensor`不能直接调用，由编译器在赋值形式中自动声明。<br>- 寄存器在`@pl.vector_function`函数内创建和使用，函数结束后自动释放。<br>- 创建寄存器后必须通过`vf.load_align`或`vf.full`初始化数据，否则内容未定义。<br>- RegTensor寄存器数量上限为32。超出限制上限的寄存器数据会写入预留的8K UB内存中，可能会引起性能劣化。编译器会自动复用生命周期结束的寄存器和预留内存，若寄存器与预留内存均存在可用空间，将优先复用寄存器。 |

## 约束说明

- 数据类型约束：

  RegTensor支持的数据类型由`dtype`参数决定，不同dtype对应不同的元素个数（寄存器总大小固定为256字节）：

  | dtype | 元素宽度 | 元素个数 |
  |---|---|---|
  | DT_INT8 / DT_UINT8 / DT_HF8 / DT_FP8E4M3FN / DT_FP8E5M2 / DT_FP8E8M0 | 8 bit | 256 |
  | DT_FP4 / DT_FP4E2M1 / DT_FP4E1M2 | 4 bit（b8 打包存储，2 元素/字节） | 256（逻辑 512） |
  | DT_INT16 / DT_UINT16 / DT_FP16 / DT_BF16 | 16 bit | 128 |
  | DT_INT32 / DT_UINT32 / DT_FP32 | 32 bit | 64 |
  | DT_INT64 / DT_UINT64 | 64 bit | 32 |

  > **FP8/FP4 说明**：FP8 类型（`DT_FP8E4M3FN`、`DT_FP8E5M2`、`DT_FP8E8M0`、`DT_HF8`）为 8 位浮点存储类型，仅支持数据搬运（`load_align`/`store_align`）、数据填充（`full`）和类型转换（`astype`），不支持直接参与算术运算。FP4 类型（`DT_FP4E2M1`、`DT_FP4E1M2`、`DT_FP4`）为 4 位浮点存储类型，两个元素打包在一个字节中（b8 存储），同样仅支持搬运、填充和类型转换。使用时需通过`vf.astype`转换为 FP32/BF16/FP16 进行计算。

## 返回值说明

`vf.reg_tensor`为类型声明，不产生返回值。寄存器由赋值形式自动声明（如`reg = vf.load_align(...)`或`reg = vf.add(...)`）。

## 关键特性

### DT_FP16类型双寄存器模式存储结构

下图为DT_FP16在单寄存器模式和双寄存器模式场景下RegTensor存储情况：

**图1**RegTensor搬运DT_FP16

![](../../../figures/reg_tensor_move_complex32.jpg "RegTensor搬运DT_FP16")

DT_FP16为16位浮点类型。在双寄存器模式场景下，从UB中以DIST_DINTLV_B16双搬入模式读取2*VL数据量，将数据交错搬运，偶数索引的元素存入reg[0]，奇数索引的元素存入reg[1]，数据类型为DT_UINT16。两个RegTensor存储512B的数据量，reg[0]和reg[1]各存128个DT_FP16元素。

### DT_FP32类型双寄存器模式存储结构

下图为DT_FP32在单寄存器模式和双寄存器模式场景下RegTensor存储情况：

**图2**RegTensor搬运DT_FP32

![](../../../figures/reg_tensor_move_complex64.jpg "RegTensor搬运DT_FP32")

DT_FP32为32位浮点类型。在双寄存器模式场景下，从UB中以DIST_DINTLV_B32双搬入模式读取2*VL数据量，将数据交错搬运，偶数索引的元素存入reg[0]，奇数索引的元素存入reg[1]，数据类型为DT_UINT32。两个RegTensor存储512B的数据量，reg[0]和reg[1]各存64个DT_FP32元素。

### DT_INT64/DT_UINT64类型双寄存器模式存储结构

下图为b64（DT_INT64、DT_UINT64）在单寄存器模式和双寄存器模式场景下RegTensor存储情况：

**图3**RegTensor搬运b64

![](../../../figures/reg_tensor_move_b64.jpg "RegTensor搬运b64")

在单寄存器模式场景下，从UB中以DIST_NORM模式搬运VL数据量。

在双寄存器模式场景下，从UB中以DIST_DINTLV_B32双搬入模式读取2*VL数据量，将b64数据交错搬运，偶数索引（低位）的元素存入reg[0]，奇数索引（高位）的元素存入reg[1]，数据类型为DT_UINT32。两个RegTensor存储512B的数据量，reg[0]存的是64个b64的前32位（低位），reg[1]存的是64个b64的后32位（高位）。

## 使用模式

### 从UB tile加载数据

```python
reg = vf.load_align(src_tile, 0)  # 从 tile 的元素个数偏移 0 处加载
```

### 作为VF运算的输入/输出

```python
preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)

reg_a = vf.load_align(src_tile, 0)
reg_b = vf.load_align(src_tile, 0)
reg_out = vf.add(reg_a, reg_b, preg)
```

### 写回UB tile

```python
vf.store_align(dst_tile, reg_out, preg)
```

### 标量初始化

```python
reg = vf.full(0.0, preg, dtype=pl.DT_FP32)  # 所有元素初始化为 0.0
```

## 与掩码寄存器的关系

`vf.reg_tensor`存储数据，掩码寄存器（由`vf.create_mask`或`vf.compare`产生）控制哪些元素参与运算：

```python
# 数据寄存器
# 掩码寄存器（控制元素范围）
preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)

reg_a = vf.load_align(src_tile, 0)
reg_b = vf.load_align(src_tile, 0)
# 只有 preg 为真的元素参与加法
reg_out = vf.add(reg_a, reg_b, preg)
```

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
    reg_out = vf.add(reg_a, reg_b, preg)
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
    torch.testing.assert_close(out, a + b, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    test_example()
    print("PASSED")
```

## 注意事项

1. **寄存器生命周期**：`vf.reg_tensor`在`@pl.vector_function`函数内创建和使用，函数结束后自动释放
2. **元素个数**：寄存器覆盖的元素个数由`dtype`决定，与`vf.create_mask`的`dtype`参数对应
3. **无需import**：`vf`是`@pl.vector_function`函数内的保留命名空间，直接使用`vf.reg_tensor`即可
4. **数据加载**：创建寄存器后必须通过`vf.load_align`或`vf.full`初始化数据，否则内容未定义
