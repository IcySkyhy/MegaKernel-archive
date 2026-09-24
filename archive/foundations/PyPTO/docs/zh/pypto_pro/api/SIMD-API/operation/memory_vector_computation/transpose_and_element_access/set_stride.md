# pypto_pro.language.set_stride

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

更新GM Tensor的各维stride。后续`load`和`store`按更新后的元素步长访问该Tensor。

常用于从GM中按非连续行间距聚集读取数据：通过将行stride设置为运行时传入的步长值，单次`load`即可把GM中间隔排列的若干行搬入同一块UB tile。

## 函数原型

```python
pypto_pro.language.set_stride(tensor, stride)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `tensor` | 输入/输出 | 要更新stride的GM Tensor |
| `stride` | 输入 | 各维元素步长序列 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `tensor` | 输入/输出 | kernel参数中的GM Tensor；stride在原Tensor上就地更新 |
| `stride` | 输入 | 长度为2的列表`[row_stride, col_stride]`，单位为Tensor元素<br>元素可为整型常量或运行时标量（如通过`getval`或Tensor下标从GM读取）<br>`col_stride`通常为`1`（行内连续）；`row_stride`大于等于列数时实现跨行聚集 |

## 流水类型

S（标量流水）。该接口仅更新Tensor的stride描述符，不读写Tile缓冲区数据，不需要buffer mutex或跨流水同步。

## 调用示例

下面是一个完整kernel：GM Tensor `x`形状为`[N_LINES, LINE]`，运行时从`strides`中读取行步长`s`，用`pypto_pro.language.set_stride`将`x`的行stride更新为`s`，随后单次`load`把第0行和第`s/LINE`行聚集搬入一块`[2, LINE]`的UB tile。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl

N_LINES = 512
LINE = 128


@pl.jit(auto_mutex=True)
def set_stride_basic_kernel(
    x: pl.Tensor[[N_LINES, LINE], pl.DT_FP16],
    strides: pl.Tensor[[1, 1], pl.DT_INT32],
    out: pl.Tensor[[2, LINE], pl.DT_FP16],
):
    ub_type = pl.TileType(shape=[2, LINE], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    ub_db = pl.make_tile_group(type=ub_type, addrs=0x0000, mutex_ids=[0, 1])

    with pl.section_vector():
        s = strides[0, 0]
        tile = ub_db.next()
        pl.set_stride(x, [s, 1])
        pl.load(tile, x, [0, 0])
        pl.store(out, tile, [0, 0])
```

配合循环可批量聚集多组行：

```python
# 此批量形式要求 strides 的 shape 为 [1, N_PAIRS]
with pl.section_vector():
    for i in pl.range(0, N_PAIRS, 1):
        s = strides[0, i]
        tile = ub_db.next()
        pl.set_stride(x, [s, 1])
        pl.load(tile, x, [i, 0])
        pl.store(out, tile, [2 * i, 0])
```
