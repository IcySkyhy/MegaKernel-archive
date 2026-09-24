# pypto_pro.language.getval

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

读取Tile或Tensor指定位置的单个元素值，返回标量。操作走标量pipe（`PipeType.S`），需要配合标量pipe同步。

统一接口：根据第一个参数的类型（Tile或Tensor）自动分发到对应的后端实现。

## 函数原型

```python
# 方式一：下标语法糖（推荐）
value = container[i]              # 1D 容器
value = container[i, j]           # 多维容器（索引数 = rank）

# 方式二：线性偏移 API
value = pypto_pro.language.getval(container, offset)
```

> **推荐使用下标语法糖**`container[i, j]`替代`pl.getval(container, offset)`。下标语法自动将多维坐标线性化为偏移量，语义更清晰。线性偏移API适用于需要直接计算线性地址的场景（如跨rank共享helper）。

## 参数类型

### 下标语法`container[i, j]`

| 参数        | 输入/输出 | 说明                                                                       |
| ----------- | --------- | -------------------------------------------------------------------------- |
| `container` | 输入      | 目标Tile或Tensor，从中读取单个元素                                      |
| `i, j, ...` | 输入      | 多维索引（整数），索引数必须等于容器rank；1D容器可用单索引`container[i]` |

### 线性偏移API `getval(container, offset)`

| 参数        | 输入/输出 | 说明                                  |
| ----------- | --------- | ------------------------------------- |
| `container` | 输入      | 目标Tile或Tensor，从中读取单个元素 |
| `offset`    | 输入      | 线性元素偏移，指定读取位置            |

## 参数范围

| 参数           | 输入/输出 | 说明                                                                                                                                                                                                                         |
| -------------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `container`    | 输入      | Tile必须位于Vec空间；Tensor为GM Tensor。支持可参与标量表达式的整型或浮点类型；不支持FP4、FP8E4M3FN、FP8E5M2、INT4、UINT4、HF4和HF8等仅用于存储的低精度类型 |
| 索引 /`offset` | 输入      | 整型常量或运行时整型标量表达式（支持循环变量）<br>下标语法：多维索引数须等于容器rank，自动线性化为`i * (N1*N2*...) + j * (N2*...) + ...`<br>线性偏移API：`offset`为线性元素偏移（0 ≤ offset < 总元素数），越界行为不确定 |

## 返回值说明

返回与`container`元素类型一致的标量。

## 流水类型

S（标量流水）。使用`make_tile_group + auto_mutex`时由框架完成MTE2→S / S→MTE3流水同步；使用`make_tile`时需显式同步。

## 调用示例

下面通过Tile和Tensor两种场景演示元素读取的用法。

### Tile场景

用下标语法读出Tile第0个元素，再写到第1个位置，store回GM验证。示例使用`make_tile_group`管理Tile资源，并通过`auto_mutex`完成MTE2→S和S→MTE3流水同步。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def getval_setval_kernel(
    a: pl.Tensor[[64, 128], pl.DT_FP16],
):
    tile_a_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[0])
    with pl.section_vector():
        tile_a = tile_a_group.current()
        pl.load(tile_a, a, [0, 0])
        value = tile_a[0, 0]      # 读 Tile[0,0] 元素
        tile_a[0, 1] = value      # 写到 Tile[0,1] 位置
        pl.store(a, tile_a, [0, 0])
```

### Tensor场景

直接从Tensor中读取标量值，写到另一个位置。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def tensor_getval_setval_kernel(
    scale_tensor: pl.Tensor[[2], pl.DT_FP32],
):
    scale = scale_tensor[0]
    scale_tensor[1] = scale
```
