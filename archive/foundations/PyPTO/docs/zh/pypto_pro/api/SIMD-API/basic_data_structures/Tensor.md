# pypto_pro.language.Tensor

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

全局内存（GM）中的多维张量类型标注，是kernel的输入/输出参数类型。

`pypto_pro.language.Tensor`主要用于：

1. kernel函数签名中声明GM张量参数
2. 配合[`pypto_pro.language.load`](../operation/memory_data_movement/load.md)/[`pypto_pro.language.store`](../operation/memory_data_movement/store.md)做GM↔UB搬运
3. 通过`=`赋值创建别名，与原变量共享同一段GM内存；支持链式别名，创建别名后重新绑定原变量不会改变已有别名的指向

## 函数原型

```python
pypto_pro.language.Tensor.__init__(
    self,
    shape: Sequence[int | _ShapePolicy | EllipsisType] | None = None,
    dtype: DataType | None = None,
    expr: Expr | None = None,
    layout: TensorLayout | None = None,
    memref: MemRef | None = None,
    _annotation_only: bool = False,
)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `shape` | 输入 | 各维大小列表 |
| `dtype` | 输入 | 元素数据类型 |
| `layout` | 输入 | 可选，内存布局 |
| `memref` | 输入 | 可选，显式内存引用；三参数形式中第三项为`MemRef`实例时按`memref`解析 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `shape` | 输入 | 维度列表<br>固定维度：正整数，如`[64, 128]`<br>动态维度：`pypto_pro.language.DYNAMIC`<br>编译期特化维度：`pypto_pro.language.STATIC`<br>末尾`...`：其余维度均按`STATIC`处理<br>不同策略可混用，如`[64, pl.DYNAMIC, pl.STATIC]` |
| `dtype` | 输入 | [`pypto_pro.language.DataType`](DataType.md)枚举值<br>常用：`pypto_pro.language.DT_FP16`、`pypto_pro.language.DT_FP32`、`pypto_pro.language.DT_BF16`、`pypto_pro.language.DT_INT8`、`pypto_pro.language.DT_INT32`、`pypto_pro.language.DT_HF8` |
| `layout` | 输入 | [`pypto_pro.language.TensorLayout`](TensorLayout.md)枚举值或`None`（默认）<br>`pypto_pro.language.ND`（非分形行主序）/ `pypto_pro.language.DN`（非分形DN布局标记）/ `pypto_pro.language.NZ`（NZ分形布局）<br>不指定时为`None`（后端按`pypto_pro.language.ND`处理） |
| `memref` | 输入 | `pypto_pro.language.MemRef`实例；需要同时指定`layout`和`memref`时使用四参数形式 |

## shape维度策略

| shape写法 | 维度来源 | 编译行为 |
|---|---|---|
| 正整数，如`128` | 类型标注中固定 | 调用时对应维度须等于该整数 |
| `pl.DYNAMIC` | 调用时读取实际维度 | 维度值不参与编译缓存键，不同取值复用同一编译变体 |
| `pl.STATIC` | 调用时读取实际维度 | 维度值固化到当前编译变体；取值变化时生成新的编译变体 |
| 末尾`...` | 调用时展开剩余维度 | 展开的各维均按`pl.STATIC`处理 |

kernel内可通过`tensor.shape[i]`读取对应维度。固定整数和绑定后的`pl.STATIC`在当前编译变体中是编译期常量，`pl.DYNAMIC`保留为运行时维度。

## 调用示例

### 类型声明

```python
import pypto_pro.language as pl

# 固定整数维度
x: pl.Tensor[[64, 128], pl.DT_FP16]

# 带布局的 tensor
y: pl.Tensor[[64, 128], pl.DT_FP16, pl.NZ]

# A矩阵的E8M0分组缩放因子：逻辑shape为[M,G]=[64,4]，GM物理shape为[M,G/2,2]=[64,2,2]
scale_a: pl.Tensor[[64, 2, 2], pl.DT_FP8E8M0]

# 动态维度声明（仅用于类型标注）
dynamic_tensor: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32]
```

### Tensor别名

Tensor别名使用普通赋值创建。赋值不会复制数据，别名可用于`load`、`store`等接收Tensor的操作。以下代码为kernel函数体内的使用片段，其中`input_tensor`、`replacement_tensor`为Tensor参数，`input_tile`、`replacement_tile`为已创建的Tile：

```python
# 一级别名和链式别名均指向首次传入的 input_tensor
original_input_alias = input_tensor
original_input_alias_chain = original_input_alias

# 别名可作为 Tensor 操作数
pl.load(input_tile, original_input_alias_chain, [0, 0])

# 重新绑定原变量不改变已有别名的指向
input_tensor = replacement_tensor
pl.load(input_tile, original_input_alias, [0, 0])  # 仍从首次传入的 input_tensor 读取
pl.load(replacement_tile, input_tensor, [0, 0])     # 从 replacement_tensor 读取
```

### DYNAMIC动态维度

以下完整kernel使用动态维度完成单tile加法。

```python
import pypto_pro.language as pl

@pl.jit(auto_mutex=True)
def dynamic_tensor_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tt = pl.TileType(shape=[64, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tile_a_group = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
    tile_b_group = pl.make_tile_group(type=tt, addrs=0x4000, mutex_ids=[1])
    tile_out_group = pl.make_tile_group(type=tt, addrs=0x8000, mutex_ids=[2])
    with pl.section_vector():
        tile_a = tile_a_group.current()
        tile_b = tile_b_group.current()
        tile_out = tile_out_group.current()
        pl.load(tile_a, a, [0, 0])
        pl.load(tile_b, b, [0, 0])
        pl.add(tile_out, tile_a, tile_b)
        pl.store(out, tile_out, [0, 0])
```

### STATIC编译期特化维度

`pl.STATIC`维度会按调用时的实际值进行编译期特化。首次出现一组新的`pl.STATIC`维度值时生成编译变体；后续调用的`pl.STATIC`维度值相同时复用已有变体，任一取值变化时生成新的变体。

```python
import pypto_pro.language as pl

TILE_M = 128
TILE_N = 128


@pl.jit(auto_mutex=True)
def add_static(
    x: pl.Tensor[[pl.STATIC, pl.STATIC], pl.DT_FP16],
    y: pl.Tensor[[pl.STATIC, pl.STATIC], pl.DT_FP16],
    z: pl.Tensor[[pl.STATIC, pl.STATIC], pl.DT_FP16],
):
    tile_type = pl.TileType(shape=[TILE_M, TILE_N], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_db = pl.make_tile_group(type=tile_type, addrs=0x0000, mutex_ids=[0, 1])
    b_db = pl.make_tile_group(type=tile_type, addrs=0x10000, mutex_ids=[2, 3])
    c_db = pl.make_tile_group(type=tile_type, addrs=0x20000, mutex_ids=[4, 5])

    with pl.section_vector():
        num_cores = pl.get_block_num()
        core_id = pl.get_block_idx()
        m_tile_num = x.shape[0] / TILE_M
        n_tile_num = x.shape[1] / TILE_N

        for i in pl.range(core_id, m_tile_num, num_cores):
            for j in pl.range(0, n_tile_num, 1):
                tile_a = a_db.next()
                tile_b = b_db.next()
                tile_c = c_db.next()
                pl.load_tile(tile_a, x, [i, j])
                pl.load_tile(tile_b, y, [i, j])
                pl.add(tile_c, tile_a, tile_b)
                pl.store_tile(z, tile_c, [i, j])
```
