# pypto_pro.language.Ptr

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

指向某种元素类型的全局内存裸指针类型标注，对应PTO MLIR的`!pto.ptr<dtype>`。

`pypto_pro.language.Ptr`主要用于：

1. kernel函数签名中声明GM裸指针参数
2. 配合[`pypto_pro.language.make_ptr`](../operation/resource_management/make_ptr.md)创建不同元素类型的指针视图
3. 配合[`pypto_pro.language.addptr`](../operation/resource_management/addptr.md)做指针偏移
4. 配合[`pypto_pro.language.make_tensor`](../operation/resource_management/make_tensor.md)从裸指针构造tensor view

## 函数原型

```python
pypto_pro.language.Ptr[dtype]
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dtype` | 输入 | 指针指向的元素数据类型 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `dtype` | 输入 | [`pypto_pro.language.DataType`](DataType.md)枚举值<br>常用：`pypto_pro.language.DT_FP16`、`pypto_pro.language.DT_FP32`、`pypto_pro.language.DT_INT8` |

## 调用示例

下面是一个完整kernel：在kernel签名中用`pypto_pro.language.Ptr`声明workspace裸指针参数，配合`addptr`偏移到后半段、`make_tensor`包装成tensor view，完成`a*2`写回`out`。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

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
