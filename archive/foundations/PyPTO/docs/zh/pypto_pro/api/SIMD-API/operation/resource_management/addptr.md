# pypto_pro.language.addptr

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

对一个裸指针（`pypto_pro.language.Ptr[dtype]`）做偏移运算，得到指向同一片GM、起点平移后的新指针。偏移以**元素**为单位（不是字节），元素大小由指针的dtype决定。

常用于把一块workspace（GM暂存区）按需切成多段，配合[`pypto_pro.language.make_tensor`](make_tensor.md)把每段包装成可load/store的Tensor视图。

## 函数原型

```python
pypto_pro.language.addptr(ptr, offset) -> Ptr
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `ptr` | 输入 | 裸指针，类型为`pypto_pro.language.Ptr[dtype]`（PtrType） |
| `offset` | 输入 | 元素偏移量 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `ptr` | 输入 | 须为`pypto_pro.language.Ptr[dtype]`标注的裸指针；返回的新指针与入参dtype相同。元素位宽必须不少于8 bit；`DT_INT4`、`DT_UINT4`、`DT_FP4`、`DT_HF4`等亚字节类型不支持指针算术，可先通过`make_ptr`重解释为`DT_UINT8`等字节可寻址类型。 |
| `offset` | 输入 | 整型常量或运行时整型标量表达式，单位为**元素**（实际字节偏移 = `offset × dtype字节数`，由编译器换算）；偏移后须仍落在原workspace范围内 |

## 调用示例

下面是一个完整Kernel：用`pypto_pro.language.addptr`将workspace裸指针偏移到后半段，配合`make_tensor`包装成Tensor视图作为暂存区，完成`a*2`写回`out`。Vector Kernel开启`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def workspace_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    workspace: pl.Ptr[pl.DT_FP16],
    out: pl.Tensor[[64, 128], pl.DT_FP16],
):
    ws_buf_ptr = pl.addptr(workspace, 64 * 128)
    ws_buf = pl.make_tensor(ws_buf_ptr, [64, 128], [128, 1])

    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])

    with pl.section_vector():
        t = tile.current()
        pl.load(t, a, [0, 0])
        pl.add(t, t, t)
        pl.store(ws_buf, t, [0, 0])
        pl.load(t, ws_buf, [0, 0])
        pl.store(out, t, [0, 0])
```
