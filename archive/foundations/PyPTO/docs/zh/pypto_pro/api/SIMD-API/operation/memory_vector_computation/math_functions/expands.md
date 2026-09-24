# pypto_pro.language.expands

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

将Tile填充为一个标量值（标量填充 / splat）。常用于初始化负无穷Tile（因果掩码）或零Tile。

## 函数原型

```python
pypto_pro.language.expands(out, scalar)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 目标Tile，全部元素被填充为`scalar` |
| `scalar` | 输入 | 填充值 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 支持8/16/32/64位整型、FP16、BF16和FP32；位于Vec或Mat空间，Vec Tile使用行主序布局<br>shape：二维Tile |
| `scalar` | 输入 | 整型或浮点型常量，或运行时整型或浮点型标量表达式<br>类型须与`out`元素类型兼容 |

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整Kernel：用`pypto_pro.language.expands`把FP32 Tile填充为标量`2.0`后写回GM。Vector Kernel开启`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl

K_VALUE = 2.0


@pl.jit(auto_mutex=True)
def expands_kernel(dummy: pl.Tensor[[64, 64], pl.DT_FP32],
                   out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_out = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    with pl.section_vector():
        cur_out = tile_out.current()
        pl.expands(cur_out, K_VALUE)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:expands:start -->
```bash
输入数据value：2
输出数据out：[[2 2 2 2 2 2 2 2 ...], [2 2 2 2 2 2 2 2 ...], [2 2 2 2 2 2 2 2 ...], [2 2 2 2 2 2 2 2 ...], ...]
```
<!-- pypto-doc-output:expands:end -->

其他典型用法（节选）：

```python
# 初始化负无穷 Tile（因果掩码）
pl.expands(neg_inf_vec, NEG_INF)

# 初始化零 Tile
pl.expands(score_u16_row, 0)
```
