# pypto_pro.language.exp

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

逐元素计算自然指数e^x。支持in-place写法。

## 函数原型

```python
pypto_pro.language.exp(out, src)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 目标tile，存放逐元素自然指数结果 |
| `src` | 输入 | 源tile |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 数据类型：FP16、FP32<br>shape须与`src`一致<br>支持与`src`为同一tile，实现in-place exp |
| `src` | 输入 | 数据类型：与`out`一致<br>shape：与`out`一致 |

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整kernel：把FP32源tile逐元素计算自然指数后写回GM。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def exp_kernel(a: pl.Tensor[[64, 64], pl.DT_FP32],
               out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_out = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.exp(cur_out, cur_a)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:exp:start -->
```bash
输入数据a：[[-4 -3.875 -3.75 -3.625 -3.5 -3.375 -3.25 -3.125 ...], [4 4.125 4.25 4.375 4.5 4.625 4.75 4.875 ...], [12 12.125 12.25 12.375 12.5 12.625 12.75 12.875 ...], [20 20.125 20.25 20.375 20.5 20.625 20.75 20.875 ...], ...]
输出数据out：[[0.018316 0.020754 0.023518 0.026649 0.030197 0.034218 0.038774 0.043937 ...], [54.598148 61.867809 70.105408 79.439842 90.017128 102.002769 115.584282 130.974152 ...], [1.627548e+05 1.844253e+05 2.089813e+05 2.368068e+05 2.683373e+05 3.040660e+05 3.445519e+05 3.904284e+05 ...], [4.851652e+08 5.497642e+08 6.229645e+08 7.059112e+08 7.999022e+08 9.064079e+08 1.027095e+09 1.163851e+09 ...], ...]
```
<!-- pypto-doc-output:exp:end -->

其他典型用法（节选）：

```python
# in-place exp
pl.exp(exp_corr_rm, exp_corr_rm)
```
