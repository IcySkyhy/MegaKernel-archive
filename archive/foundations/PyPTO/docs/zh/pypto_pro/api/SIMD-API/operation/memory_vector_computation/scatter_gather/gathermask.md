# pypto_pro.language.gathermask

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

按位模式从源tile中抽取列到目标tile。

`pattern_mode`取值：

| 取值 | 含义 |
|---|---|
| 1 | 取偶数列`src[:, 0::2]` |
| 2 | 取奇数列`src[:, 1::2]` |
| 3 | 每4列取第1列`src[:, 0::4]` |
| 4 | 每4列取第2列`src[:, 1::4]` |
| 5 | 每4列取第3列`src[:, 2::4]` |
| 6 | 每4列取第4列`src[:, 3::4]` |
| 7 | 全取（等价于copy） |

## 函数原型

```python
pypto_pro.language.gathermask(out, src, *, pattern_mode)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 目标tile，存放按位模式抽取的列 |
| `src` | 输入 | 源tile |
| `pattern_mode` | 输入 | 位模式（常量整数），决定抽取哪些列 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 数据类型：与`src`一致<br>shape：行数与`src`一致，列数由`pattern_mode`决定 |
| `src` | 输入 | 数据类型：b8、b16、b32、b64，`out`与`src`元素位宽相同；二者均为Vec、行主序Tile<br>`out`的有效列必须连续存储 |
| `pattern_mode` | 输入 | 编译期常量整数，取值1～7<br>1/2：每2列分别取第1/2列<br>3/4/5/6：每4列分别取第1/2/3/4列<br>7：取全部列 |

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整kernel：用`pypto_pro.language.gathermask`从64×128 FP16源tile中抽取偶数列（`pattern_mode=1`），输出64×64 FP16 tile。纯vector kernel使用`make_tile_group`管理Tile资源，并通过`auto_mutex`完成流水同步。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def gathermask_p1_kernel(
    src: pl.Tensor[[64, 128], pl.DT_FP16],
    dst: pl.Tensor[[64, 64], pl.DT_FP16],
):
    tile_src_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x0000, mutex_ids=[0])
    tile_dst_group = pl.make_tile_group(
        type=pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec),
        addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        tile_src = tile_src_group.current()
        tile_dst = tile_dst_group.current()
        pl.load(tile_src, src, [0, 0])
        pl.gathermask(tile_dst, tile_src, pattern_mode=1)
        pl.store(dst, tile_dst, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:gathermask:start -->
```bash
输入数据src：[[1 1.25 1.5 1.75 2 2.25 2.5 2.75 ...], [33 33.25 33.5 33.75 34 34.25 34.5 34.75 ...], [65 65.25 65.5 65.75 66 66.25 66.5 66.75 ...], [97 97.25 97.5 97.75 98 98.25 98.5 98.75 ...], ...]
输入数据pattern_mode：1
输出数据dst：[[1 1.5 2 2.5 3 3.5 4 4.5 ...], [33 33.5 34 34.5 35 35.5 36 36.5 ...], [65 65.5 66 66.5 67 67.5 68 68.5 ...], [97 97.5 98 98.5 99 99.5 100 100.5 ...], ...]
```
<!-- pypto-doc-output:gathermask:end -->

其他典型用法（节选）：

```python
# 抽取奇数列
pl.gathermask(tile_dst, tile_src, pattern_mode=2)

# 全取（copy）
pl.gathermask(tile_dst, tile_src, pattern_mode=7)
```
