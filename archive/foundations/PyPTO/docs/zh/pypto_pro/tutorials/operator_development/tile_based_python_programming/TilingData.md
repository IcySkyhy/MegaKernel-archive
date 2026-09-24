# TilingData

本节介绍TilingData的声明、传入和字段访问方法。TilingData用于向已编译Kernel传递
shape、stride、循环边界、算子选择器和缩放系数等运行时参数，无需将具体取值固化在
Kernel签名中。TilingData既支持标量字段，也支持定长数组字段。

示例代码使用以下导入：

```python
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch_npu
import pypto_pro.language as pl
```

---

## TilingData类

TilingData类是一个普通的Python `@dataclass`，其字段**全部**为以下之一：

- `int`  —— 标量整数  （降低为`INDEX` / `int64_t`）
- `float`—— 标量浮点  （降低为`FP32`）
- `bool` —— 标量布尔  （降低为`BOOL`）
- `T[N]` —— 定长数组，含`N`个`T`类型元素（`T` ∈ `{int, float, bool}`）

TilingData类至少包含一个字段，并且每个字段均使用上述类型之一进行标注。

字段支持Python `dataclass`默认值。框架序列化实例中的全部字段值，未显式传入的字段
使用`dataclass`默认值。设备侧结构体包含全部字段。数组字段需要使用
`dataclasses.field(default_factory=...)`提供默认列表，
且带默认值的字段必须位于无默认值字段之后。

将TilingData实例传给JIT Kernel时，框架会：

1. 根据字段标注推导C `struct`布局；
2. 按照该布局序列化实例，并传递给设备侧代码。

字段类型与C struct成员一一对应（采用原生ctypes对齐），因此Python侧与生成的CCE
struct在`sizeof`与字段偏移上保持一致。

```python
TS = 128       # Q方向的Tile大小，编译期常量
TKV = 128      # K/V方向的Tile大小，编译期常量


@dataclass
class OpTiling:
    sq: int       # 标量
    skv: int      # 标量
    d: int        # 标量
```

上述代码定义了一个包含三个整型标量字段的TilingData类。

---

## 在Kernel签名中声明TilingData

在Kernel函数形参末尾声明TilingData类型参数。该参数按运行时结构体传递，可与裸指针
输入（[`pl.Ptr[dtype]`](../../../api/SIMD-API/basic_data_structures/Ptr.md)）配合，使用TilingData中的shape重建固定rank的Tensor视图：

```python
@dataclass
class OpTiling:
    sq: int
    skv: int
    d: int

@pl.jit(auto_mutex=True)
def fa_kernel(
    q: pl.Ptr[pl.DT_FP16],
    k: pl.Ptr[pl.DT_FP16],
    v: pl.Ptr[pl.DT_FP16],
    o: pl.Ptr[pl.DT_FP16],
    tiling: OpTiling,          # <-- TilingData 参数
):
    # 重建二维带类型视图；两个维度的运行时取值来自 tiling，而非函数签名。
    tensor_q = pl.make_tensor(q, [tiling.sq,  tiling.d])
    tensor_k = pl.make_tensor(k, [tiling.skv, tiling.d])
    tensor_v = pl.make_tensor(v, [tiling.skv, tiling.d])
    tensor_o = pl.make_tensor(o, [tiling.sq,  tiling.d])

    sq_dim    = tiling.sq
    skv_dim   = tiling.skv
    sq_tiles  = (sq_dim  + (TS  - 1)) // TS
    skv_tiles = (skv_dim + (TKV - 1)) // TKV
    # Kernel中将tiling字段作为普通运行时标量使用
```

上述示例使用[`pl.make_tensor`](../../../api/SIMD-API/operation/resource_management/make_tensor.md)从裸指针构造固定rank的Tensor视图。也可以保留带类型的[`pl.Tensor`](../../../api/SIMD-API/basic_data_structures/Tensor.md)输入，并仅使用TilingData传递循环边界、标志和缩放系数。
两种用法相互独立。

> [!NOTE]说明
> 当前每次Kernel调用仅支持一个TilingData实例，且该实例必须位于Kernel形参列表和
> 启动实参列表的末尾。

---

## 在Kernel中读取TilingData字段

在Kernel中，TilingData字段是普通的运行时值：

| 访问形式             | 含义                                                |
|----------------------|-----------------------------------------------------|
| `tiling.field`       | 读取标量字段（int / float / bool）。               |
| `tiling.arr[k]`      | 读取`T[N]`字段的第`k`个元素。           |

它们可用于：

- `pl.make_tensor`的shape和stride实参，
- `pl.range(...)`的循环边界，
- 算术运算（`(tiling.sq + TS - 1) // TS`），
- 运行时条件分支（`if tiling.opkind[4] == 0:`）。

```python
# 标量字段用作循环边界
for kv in pl.range(0, skv_tiles, 1):
    ...

# 数组元素用于在运行时选择算子
if tiling.opkind[4] == 0:
    pl.add(tile_c, tile_a, tile_b)
elif tiling.opkind[4] == 1:
    pl.sub(tile_c, tile_a, tile_b)
else:
    pl.mul(tile_c, tile_a, tile_b)
```

---

## 在启动时构造并传入TilingData实例

在Host侧构造dataclass，并通过方括号启动语法作为对应参数传入。标量字段可直接传值：

```python
tiling = OpTiling(sq=8192, skv=8192, d=128)
fa_kernel[None, num_cores](q_t, k_t, v_t, o_t, tiling)
torch.npu.synchronize()
```

框架自动将`tiling`序列化为C struct字节并管理相应字节缓冲。

![PyPTO Pro TilingData从Host实例到Kernel字段访问的传递流程](../../figures/pro_tilingdata_host_kernel_flow.png)

运行时按照字段定义顺序和原生ctypes对齐规则生成C struct字节，再把它包装为设备侧
`uint8` 缓冲区传给Kernel。生成的CCE结构体与Host侧在`sizeof`和字段偏移上保持一致。

---

## 数组字段

数组字段持有固定数量的同质元素。用`T[N]`声明字段，用Python原生`list`
保存运行时值。包含`T[N]`的模块必须启用`from __future__ import annotations`，
避免Python在定义class时求值`int[N]`。

### 声明数组字段

```python
@dataclass
class OpTiling:
    offsets: int[4]     # 4 个 int
    scales:  float[2]   # 2 个 float
    opkind:  int[8]     # 8 个 int
```

数组字段声明需满足以下规则：

- 元素类型必须为`int`、`float`或`bool`；
- 数组长度`N`必须直接写成1～2048的整数值（例如`int[4]`），不能使用变量、
  算术表达式或布尔值。

`int[N]`是PyPTO Pro DSL注解，不是标准Python类型；部分静态类型检查器可能提示
`int`不可下标，但PyPTO Pro会从延迟注解字符串中安全解析该字段。

### 构造数组值

`T[N]`只描述字段，不负责构造值。使用原生列表即可：

```python
[0] * 4                         # -> [0, 0, 0, 0]
[0, 1, 2]                       # 直接给出元素
[i for i in range(60)]          # 由迭代生成

arr = [0] * 8                  # 可变：支持下标赋值
arr[4] = 1                     # 设置第 4 个元素
```

`list`支持`arr[i]`读/写、`len(arr)`与迭代。长度必须与声明的大小一致，
否则序列化会抛出`ValueError`。

---

## 完整、可执行示例 —— 运行时shape与尾块处理

以下示例使用两个定长数组：`shape`用于将Host侧2～4维输入的shape折叠为Kernel中的二维Tensor视图，
`opkind[4]`用于在运行时选择加、减或乘。示例同时使用`valid_shape`处理不能被
`128 × 128`整除的尾块。

以`shape=[1, 1, 513, 511]`、`opkind[4]=1`为例：同一份TilingData同时决定Tensor
视图、Tile数量、尾块有效形状和算子分支，这些值均在运行时生效。

```python
from __future__ import annotations

import os

from dataclasses import dataclass

import logging
import torch
import torch_npu
import pypto_pro.language as pl

MAX_RANK = 4
TILE_M = 128
TILE_N = 128

@dataclass
class AddTiling:
    shape: int[4]       # 未使用的前导维度填1
    opkind: int[8]      # opkind[4]保存算子选择值


@pl.jit(auto_mutex=True)
def add_dynrank_kernel(
    x: pl.Ptr[pl.DT_FP16],
    y: pl.Ptr[pl.DT_FP16],
    z: pl.Ptr[pl.DT_FP16],
    tiling: AddTiling,
):
    # 把Host侧2～4维输入的shape折叠成Kernel内固定的二维[M, N] Tensor视图。
    n = tiling.shape[3]
    m = tiling.shape[0] * tiling.shape[1] * tiling.shape[2]
    tensor_x = pl.make_tensor(x, [m, n])
    tensor_y = pl.make_tensor(y, [m, n])
    tensor_z = pl.make_tensor(z, [m, n])

    tile_type = pl.TileType(
        shape=[TILE_M, TILE_N],
        dtype=pl.DT_FP16,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
    )
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x20000, mutex_ids=[30, 31])
    with pl.section_vector():
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()
        m_tiles = (m + TILE_M - 1) // TILE_M
        n_tiles = (n + TILE_N - 1) // TILE_N
        total_tiles = m_tiles * n_tiles

        for idx in pl.range(core_id, total_tiles, num_cores):
            i = idx // n_tiles
            j = idx % n_tiles
            tile_a = a_db.next()
            tile_b = b_db.next()
            tile_c = c_db.next()

            # 分别覆盖满块、尾列、尾行和尾角四种情况。
            rem_r = m - i * TILE_M
            rem_c = n - j * TILE_N
            if rem_r >= TILE_M:
                if rem_c >= TILE_N:
                    pl.set_validshape(tile_a, [TILE_M, TILE_N])
                    pl.set_validshape(tile_b, [TILE_M, TILE_N])
                    pl.set_validshape(tile_c, [TILE_M, TILE_N])
                else:
                    pl.set_validshape(tile_a, [TILE_M, rem_c])
                    pl.set_validshape(tile_b, [TILE_M, rem_c])
                    pl.set_validshape(tile_c, [TILE_M, rem_c])
            else:
                if rem_c >= TILE_N:
                    pl.set_validshape(tile_a, [rem_r, TILE_N])
                    pl.set_validshape(tile_b, [rem_r, TILE_N])
                    pl.set_validshape(tile_c, [rem_r, TILE_N])
                else:
                    pl.set_validshape(tile_a, [rem_r, rem_c])
                    pl.set_validshape(tile_b, [rem_r, rem_c])
                    pl.set_validshape(tile_c, [rem_r, rem_c])

            pl.load_tile(tile_a, tensor_x, [i, j])
            pl.load_tile(tile_b, tensor_y, [i, j])

            if tiling.opkind[4] == 0:
                pl.add(tile_c, tile_a, tile_b)
            elif tiling.opkind[4] == 1:
                pl.sub(tile_c, tile_a, tile_b)
            else:
                pl.mul(tile_c, tile_a, tile_b)

            pl.store_tile(tensor_z, tile_c, [i, j])

OP_CASES = [
    (0, lambda a, b: a + b, "add"),
    (1, lambda a, b: a - b, "sub"),
    (2, lambda a, b: a * b, "mul"),
]


def ceildiv(a, b):
    return (a + b - 1) // b


def _run_case(shape, opkind, ref_fn, op_name):
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    torch.npu.set_device(device)
    torch.manual_seed(0)
    dtype = torch.float16

    rank = len(shape)
    assert 2 <= rank <= MAX_RANK, f"rank must be in [2, {MAX_RANK}], got {rank}"

    numel = 1
    for s in shape:
        numel *= s

    x = torch.rand(shape, device=device, dtype=dtype)
    y = torch.rand(shape, device=device, dtype=dtype)
    z = torch.empty(shape, device=device, dtype=dtype)

    # 使用前导1补齐到长度4，使最内层维度始终位于shape[3]。
    dims = [1] * MAX_RANK
    for i in range(rank):
        dims[MAX_RANK - rank + i] = shape[i]

    opkind_arr = [0] * 8
    opkind_arr[4] = opkind
    tiling = AddTiling(shape=dims, opkind=opkind_arr)

    n = shape[-1]
    m = numel // n
    total_tiles = ceildiv(m, TILE_M) * ceildiv(n, TILE_N)
    num_cores = min(32, total_tiles)

    add_dynrank_kernel[None, num_cores](x, y, z, tiling)
    torch.npu.synchronize()

    z_ref = ref_fn(x.float(), y.float()).half()
    torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)
    logging.info("dynamic-rank %s %s (rank=%d, numel=%d) passed!", op_name, list(shape), rank, numel)


def test_add_dynamic_rank():
    shapes = [
        [512, 512],
        [8, 256, 256],
        [2, 4, 256, 256],
        [513, 513],
        [513, 511],
        [200, 300],
        [2, 3, 513],
        [2, 2, 3, 200],
    ]
    for shape in shapes:
        for opkind, ref_fn, op_name in OP_CASES:
            _run_case(shape, opkind, ref_fn, op_name)


if __name__ == "__main__":
    test_add_dynamic_rank()
```

本例演示了：

- 使用多个`int[N]`数组字段的TilingData类；
- Kernel中的**数组元素访问**（`tiling.opkind[4]`）驱动不同的计算分支；
- `tiling.shape[0..3]`（`int[4]`数组）保存Host输入的运行时shape；Kernel构造的`[M, N]` Tensor视图的rank固定为2；
- `valid_shape`和`pl.set_validshape`用于安全处理任意二维尾块；
- 同一个已编译Kernel可根据启动时传入的TilingData值运行三种不同算子，无需重新编译。

---

## 纯标量TilingData示例

仅需传递少量运行时标量时，可以使用只包含标量字段的TilingData类：

```python
@dataclass
class LoopTiling:
    n_iters: int          # 运行时循环边界

@pl.jit(auto_mutex=True)
def copy_kernel(
    x: pl.Tensor[[1024, 256], pl.DT_FP16],
    z: pl.Tensor[[1024, 256], pl.DT_FP16],
    tiling: LoopTiling,
):
    tt = pl.TileType(shape=[1, 256], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    g  = pl.make_tile_group(type=tt, addrs=0x0, mutex_ids=[0, 1])
    with pl.section_vector():
        for i in pl.range(0, tiling.n_iters, 1):     # 使用TilingData中的运行时循环边界
            buf = g.next()
            pl.load_tile(buf, x, [i, 0])
            pl.store_tile(z, buf, [i, 0])

# Host侧构造并启动
tiling = LoopTiling(n_iters=4)
copy_kernel[None, 1](x, z, tiling)
```

---

## 字段类型与dtype对应关系

TilingData字段标注与IR dtype、C struct成员的对应关系如下：

| 标注           | IR dtype | C struct成员（CCE）             |
|----------------|----------|----------------------------------|
| `int`          | `INDEX`  | `int64_t`                        |
| `float`        | `FP32`   | `float`                          |
| `bool`         | `BOOL`   | 布尔大小的整数                   |
| `int[N]`       | `INDEX`  | `int64_t[N]`                     |
| `float[N]`     | `FP32`   | `float[N]`                       |
| `bool[N]`      | `BOOL`   | 布尔大小的整数`[N]`             |

布局采用原生ctypes对齐计算，并与代码生成的C struct保持一致。字段顺序决定结构体布局，
因此Python dataclass的字段顺序必须与设备侧预期一致。

---

## 使用限制

- **不符合规范的字段类型。** 每个字段必须是`int`/`float`/`bool`或`T[N]`。
  其他类型标注不会被识别为有效的TilingData字段。
- **数组长度错误。** 赋给`int[8]`字段的序列必须恰好包含8个元素；长度不符会在
  JIT启动序列化时抛出`ValueError`。
- **数组声明过大。** `T[N]`中的`N`必须是正整数，且不能超过2048。
- **参数位置错误。** TilingData必须位于Kernel形参和启动实参的末尾。
- **字段顺序与布局不匹配。** dataclass的字段顺序决定struct布局，应与设备侧预期的
  字段顺序和填充方式保持一致。
- **运行时数据与编译期常量混用。** TilingData字段是运行时值，可用于`pl.range`、
  `pl.make_tensor`、算术和`if`条件，但不能用于需要编译期Python `int`的参数，
  例如`TileType`的静态`shape`。

---

## 使用摘要

```python
from __future__ import annotations

# 声明TilingData
@dataclass
class MyTiling:
    n: int
    scale: float
    flags: int[8]

# 在Kernel签名中声明TilingData参数
@pl.jit(auto_mutex=True)
def k(x: pl.Ptr[pl.DT_FP16], tiling: MyTiling):
    t = pl.make_tensor(x, [tiling.n, 128])             # shape 中的标量字段
    for i in pl.range(0, tiling.n, 1):                 # 作循环边界的标量字段
        if tiling.flags[4] == 1:                       # 分支中的数组元素
            ...
    ...

# 在Host侧构造并启动
flags = [0] * 8
flags[4] = 1
tiling = MyTiling(n=256, scale=2.0, flags=flags)
k[None, num_cores](x, tiling)
```
