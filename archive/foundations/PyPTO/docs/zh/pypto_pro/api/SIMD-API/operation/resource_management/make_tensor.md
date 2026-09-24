# pypto_pro.language.make_tensor

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

用一个裸指针（`pypto_pro.language.Ptr[dtype]`）或已有Tensor，加上显式的shape和stride，构造一个Tensor视图。Tensor视图本身不分配内存，只为同一段GM地址设置“形状 + 步长”的解释，之后即可像普通GM Tensor一样被[`load`](../memory_data_movement/load.md)/[`store`](../memory_data_movement/store.md)使用。省略stride时，接口根据shape自动生成连续的行主序stride。

常与[`pypto_pro.language.addptr`](addptr.md)配合：用`addptr`切出workspace的某一段地址，再用`make_tensor`把它包装成可读写的Tensor。

下图展示`make_tensor`的核心语义：它使用shape和stride为已有地址创建Tensor视图，不申请新内存，也不搬运数据。

![make_tensor从裸指针创建Tensor视图](../../../figures/make_tensor_view.jpg "make_tensor从裸指针创建Tensor视图")

## 函数原型

```python
pypto_pro.language.make_tensor(ptr, shape, stride=None, dtype=None) -> Tensor
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `ptr` | 输入 | 裸指针`pypto_pro.language.Ptr[dtype]`（PtrType）或已有Tensor；新view复用其底层地址 |
| `shape` | 输入 | 各维大小，list或MakeTuple |
| `stride` | 输入 | 可选，各维步长（单位元素），list或MakeTuple；省略时根据shape推导生成连续stride |
| `dtype` | 输入 | 可选，view的元素dtype；不传时从源指针的pointee类型或源Tensor的dtype推导 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `ptr` | 输入 | 须为`pypto_pro.language.Ptr[dtype]`标注的裸指针或已有Tensor；view与源对象共享底层数据地址 |
| `shape` | 输入 | 各维大小（整型常量或运行时整型标量表达式） |
| `stride` | 输入 | 可选，各维步长，单位为**元素**。省略时自动生成连续stride：`stride[i] = shape[i+1] * ... * shape[-1]`，且`stride[-1] = 1`。显式传入时，元素`[i0, i1, ...]`相对`ptr`的偏移为`i0 * stride[0] + i1 * stride[1] + ...`；可表达连续视图、高轴非连续视图或交换轴后的视图 |
| `dtype` | 输入 | 可选，若传入则view以该dtype创建（相当于把源地址reinterpret为`dtype`）；不传时沿用源指针或源Tensor的元素类型 |

> [!NOTE]
> `make_tensor`记录的stride同时用于`load`/`store`的起始地址换算和GlobalTensor搬运描述符。例如`[8, 16] / [32, 1]`在offset`[i, j]`处从`ptr + i * 32 + j`开始访问，跨行时也按32个元素前进。stride仍需满足目标后端和搬运接口的能力范围；CCE通用`load`/`store`主要用于尾轴连续的二维搬运。

省略stride的连续视图等价于显式传入连续stride：

```python
default_contiguous = pl.make_tensor(ptr, [8, 16])
# <=> 等价于
explicit_contiguous = pl.make_tensor(ptr, [8, 16], [16, 1])
```

## stride视图示例

下面两个场景都是对基础Tensor视图的进一步使用：左侧通过高轴stride表达行间空洞，右侧通过交换shape和stride表达转置后的逻辑索引。

![make_tensor的非连续与转置视图](../../../figures/make_tensor_stride_cases.jpg "make_tensor的非连续与转置视图")

### 高轴非连续、尾轴连续

高轴非连续是指同一行内部连续，但相邻两行的起始地址之间存在间隔。例如：

```python
pitched = pl.make_tensor(ptr, [8, 16], [32, 1])
```

其中尾轴stride为1，因此`pitched[i, 0]`到`pitched[i, 15]`对应16个连续元素；高轴stride为32，因此下一行从`ptr + (i + 1) * 32`开始，中间跳过16个元素。元素`pitched[i, j]`的地址为：

```text
ptr + i * 32 + j
```

这种视图适合表达带行间填充（Padding）或从更宽二维Tensor中截取的尾轴连续区域。

### 交换shape和stride表达转置视图

同一段连续内存可以使用不同的shape / stride解释：

```python
normal = pl.make_tensor(ptr, [8, 16]) <=> normal = pl.make_tensor(ptr, [8, 16], [16, 1])
transposed = pl.make_tensor(ptr, [16, 8], [1, 16])
```

两者共享同一个`ptr`，但逻辑轴的解释相反：

```text
normal[i, j]     -> ptr + i * 16 + j
transposed[j, i] -> ptr + j * 1 + i * 16
```

`normal[i, j]`与`transposed[j, i]`的地址相同；相同索引下的两个Tensor视图对应不同的物理元素。

## 调用示例

下面是一个完整kernel：用`addptr`切出workspace后，`make_tensor`把裸指针包装成shape为`[64, 128]`、行stride为256的非连续Tensor视图。kernel分两次处理数据，第二次使用非零offset`[32, 0]`，验证`load`和`store`都按view stride访问。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def workspace_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
    workspace: pl.Ptr[pl.DT_FP16],
    out: pl.Tensor[[64, 128], pl.DT_FP16],
):
    ws_buf_ptr = pl.addptr(workspace, 64 * 128)
    ws_buf = pl.make_tensor(ws_buf_ptr, [64, 128], [256, 1])

    tt = pl.TileType(shape=[32, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tile = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])

    with pl.section_vector():
        t = tile.current()
        pl.load(t, a, [0, 0])
        pl.add(t, t, t)
        pl.store(ws_buf, t, [0, 0])
        pl.load(t, ws_buf, [0, 0])
        pl.store(out, t, [0, 0])

        pl.load(t, a, [32, 0])
        pl.add(t, t, t)
        pl.store(ws_buf, t, [32, 0])
        pl.load(t, ws_buf, [32, 0])
        pl.store(out, t, [32, 0])
```
