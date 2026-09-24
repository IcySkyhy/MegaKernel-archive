# pypto_pro.language.fill_index

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

向目标tile填充从`start`开始的连续整数序列：`out[j] = start + j`。

典型场景：生成位置编码、初始化索引tile用于排序或gather操作。

## 函数原型

```python
pypto_pro.language.fill_index(out, start)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 目标tile，存放生成的索引序列 |
| `start` | 输入 | 起始值 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 数据类型：`pypto_pro.language.DT_INT32`<br>shape：行数为1，列数为索引个数 |
| `start` | 输入 | 整数或标量表达式 |

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整kernel：用`pypto_pro.language.fill_index`向`[1, 64]`的INT32 tile填充从`START`开始的连续整数序列，按动态shape循环store回GM。

```python
import pypto_pro.language as pl

START = 0


@pl.jit(auto_mutex=True)
def fill_index_kernel(
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_INT32],
):
    tt = pl.TileType(shape=[1, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec)
    tile_out = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    with pl.section_vector():
        m_dim = out.shape[0]
        n_dim = out.shape[1]
        for i in pl.range(0, m_dim, 1):
            for j in pl.range(0, n_dim, 64):
                cur_out = tile_out.current()
                pl.fill_index(cur_out, START)
                pl.store(out, cur_out, [i, j])
```
