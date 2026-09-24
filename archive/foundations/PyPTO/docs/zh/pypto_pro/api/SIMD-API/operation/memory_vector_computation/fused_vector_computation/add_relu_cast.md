# pypto_pro.language.add_relu_cast

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

先对两个tile做逐元素加法，再对结果施加ReLU激活（负值置零），最后做数据类型转换。当前后端依次发射`TADD`、`TRELU`和`TCVT`，并非单条硬件指令。

## 函数原型

```python
pypto_pro.language.add_relu_cast(out, lhs, rhs, *, target_type, mode=pl.RoundMode.CAST_ROUND)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 目标tile，存放先加后ReLU再类型转换的结果 |
| `lhs` | 输入 | 左操作数tile |
| `rhs` | 输入 | 右操作数tile |
| `target_type` | 输入 | 输出数据类型，如`pypto_pro.language.DT_FP32`、`pypto_pro.language.DT_FP16` |
| `mode` | 输入 | 舍入模式，默认`pl.RoundMode.CAST_ROUND` |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `out` | 输出 | 数据类型：由`target_type`指定<br>shape须与`lhs`、`rhs`一致 |
| `lhs` | 输入 | 中间加法/ReLU类型须为FP16或FP32；实现会用`lhs`保存中间结果，因此调用后其内容被覆盖<br>shape：与`out`一致 |
| `rhs` | 输入 | 数据类型：与`lhs`一致<br>shape：与`out`一致 |
| `target_type` | 输入 | 支持`pypto_pro.language.DT_FP16`、`pypto_pro.language.DT_BF16`、`pypto_pro.language.DT_FP32`等<br>可与输入类型相同（仅做ReLU）或不同（融合类型转换） |
| `mode` | 输入 | 舍入模式：`pl.RoundMode.CAST_NONE` / `pl.RoundMode.CAST_RINT` / `pl.RoundMode.CAST_ROUND` / `pl.RoundMode.CAST_FLOOR` / `pl.RoundMode.CAST_CEIL` / `pl.RoundMode.CAST_TRUNC` / `pl.RoundMode.CAST_ODD`<br>缩窄转换（如FP32→FP16）时生效，扩展转换时忽略 |

## 流水类型

V（向量计算流水）。

## 调用示例

下面是一个完整kernel：从GM载入两个FP16输入，用`pypto_pro.language.add_relu_cast`先加后ReLU再转FP32写回GM。vector kernel开`auto_mutex`，同步由`make_tile_group`自动管理。

```python
import pypto_pro.language as pl


@pl.jit(auto_mutex=True)
def add_relu_cast_kernel(a: pl.Tensor[[64, 64], pl.DT_FP16], b: pl.Tensor[[64, 64], pl.DT_FP16],
                         out: pl.Tensor[[64, 64], pl.DT_FP32]):
    tt_in = pl.TileType(shape=[64, 64], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    tt_out = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a = pl.make_tile_group(type=tt_in, addrs=0x0000, mutex_ids=[0])
    tile_b = pl.make_tile_group(type=tt_in, addrs=0x2000, mutex_ids=[1])
    tile_out = pl.make_tile_group(type=tt_out, addrs=0x4000, mutex_ids=[2])
    with pl.section_vector():
        cur_a = tile_a.current()
        cur_b = tile_b.current()
        cur_out = tile_out.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.add_relu_cast(cur_out, cur_a, cur_b, target_type=pl.DT_FP32, mode=pl.RoundMode.CAST_ROUND)
        pl.store(out, cur_out, [0, 0])
```

实测结果示例如下：

<!-- pypto-doc-output:add_relu_cast:start -->
```bash
输入数据a：[[-2 -1.75 -1.5 -1.25 -1 -0.75 -0.5 -0.25 ...], [14 14.25 14.5 14.75 15 15.25 15.5 15.75 ...], [30 30.25 30.5 30.75 31 31.25 31.5 31.75 ...], [46 46.25 46.5 46.75 47 47.25 47.5 47.75 ...], ...]
输入数据b：[[3 2.875 2.75 2.625 2.5 2.375 2.25 2.125 ...], [-5 -5.125 -5.25 -5.375 -5.5 -5.625 -5.75 -5.875 ...], [-13 -13.125 -13.25 -13.375 -13.5 -13.625 -13.75 -13.875 ...], [-21 -21.125 -21.25 -21.375 -21.5 -21.625 -21.75 -21.875 ...], ...]
输出数据out：[[1 1.125 1.25 1.375 1.5 1.625 1.75 1.875 ...], [9 9.125 9.25 9.375 9.5 9.625 9.75 9.875 ...], [17 17.125 17.25 17.375 17.5 17.625 17.75 17.875 ...], [25 25.125 25.25 25.375 25.5 25.625 25.75 25.875 ...], ...]
```
<!-- pypto-doc-output:add_relu_cast:end -->
