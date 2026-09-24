# pypto_pro.language.relu

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

逐元素ReLU激活：`out = max(src, 0)`。将源tile中的负值置零，正值保持不变。

## 函数原型

```python
pypto_pro.language.relu(out, src)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 目标tile，存放逐元素ReLU结果（负值置零） |
| `src` | 输入 | 源tile |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 数据类型：FP16、FP32或INT32<br>shape和valid_shape须与`src`一致；支持与`src`为同一tile，实现in-place ReLU |
| `src` | 输入 | 数据类型：与`out`一致<br>shape：与`out`一致 |

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整kernel：把FP32源tile逐元素ReLU后写回GM。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def relu_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32],
                out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.relu(cur_out, cur_a)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:relu:start -->
```bash
输入数据a：[[-4 -3.875 -3.75 -3.625 -3.5 -3.375 -3.25 -3.125 ...], [4 4.125 4.25 4.375 4.5 4.625 4.75 4.875 ...], [12 12.125 12.25 12.375 12.5 12.625 12.75 12.875 ...], [20 20.125 20.25 20.375 20.5 20.625 20.75 20.875 ...], ...]
输出数据out：[[0 0 0 0 0 0 0 0 ...], [4 4.125 4.25 4.375 4.5 4.625 4.75 4.875 ...], [12 12.125 12.25 12.375 12.5 12.625 12.75 12.875 ...], [20 20.125 20.25 20.375 20.5 20.625 20.75 20.875 ...], ...]
```
<!-- pypto-doc-output:relu:end -->
