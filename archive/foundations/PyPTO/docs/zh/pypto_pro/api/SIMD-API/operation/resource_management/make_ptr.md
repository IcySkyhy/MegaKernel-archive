# pypto_pro.language.make_ptr

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

从已有Tensor提取底层裸指针，或为已有裸指针创建新的元素类型视图。返回指针与源对象的底层地址相同；指定`dtype`时，返回指针按该元素类型解释。

常用于将按字节粒度（如`pl.Ptr[pl.DT_UINT8]`）申请的workspace按更宽的数据类型（如`pl.DT_FP16`）解释，再配合[`pypto_pro.language.make_tensor`](make_tensor.md)包装成可load/store的Tensor视图。`addptr`按元素偏移指针起点并保留元素类型；`make_ptr`保留指针起点并设置元素类型。

## 函数原型

```python
pypto_pro.language.make_ptr(ptr, dtype=None) -> Ptr
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `ptr` | 输入 | `pypto_pro.language.Tensor`或`pypto_pro.language.Ptr[dtype]` |
| `dtype` | 输入 | 可选的目标元素类型 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `ptr` | 输入 | Tensor或裸指针。传入Tensor时提取其底层地址；传入裸指针时复用原地址 |
| `dtype` | 输入 | [`pypto_pro.language.DataType`](../../basic_data_structures/DataType.md)枚举值或`None`<br>传入时返回的指针以该dtype解释后续指针运算（相当于reinterpret cast）<br>不传时保留源对象的元素类型 |

## 返回值

返回与源Tensor或源指针地址相同、元素类型为`dtype`的`Ptr`；省略`dtype`时保留源对象的元素类型。

## 调用示例

下面是一个完整Kernel：Kernel接收一段`pl.Ptr[pl.DT_UINT8]`的GM workspace，用`pypto_pro.language.make_ptr`将其重新解释为`pl.DT_FP16`指针，再通过`make_tensor`包装成`[64, 128]`的Tensor视图，从中load数据并store到输出Tensor。Vector Kernel开启`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def make_ptr_kernel(
    workspace: pl.Ptr[pl.DT_UINT8],
    out: pl.Tensor[[64, 128], pl.DT_FP16],
):
    fp16_ptr = pl.make_ptr(workspace, dtype=pl.DT_FP16)
    ws_buf = pl.make_tensor(fp16_ptr, [64, 128], [128, 1])

    tt = pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])

    with pl.section_vector():
        t = tile.current()
        pl.load(t, ws_buf, [0, 0])
        pl.store(out, t, [0, 0])
```

`make_ptr`不传`dtype`时等价于身份转换，返回与源指针同类型的指针：

```python
same = pl.make_ptr(ptr)
```

从Tensor提取同元素类型的裸指针：

```python
ptr = pl.make_ptr(tensor)
```
