# TilingKey

本节介绍TilingKey的声明和使用方法。TilingKey使用有限的编译期配置，为同一份
PyPTO Pro Kernel生成多个专用实例，并在启动时选择目标实例。TilingKey适用于会改变
代码路径、Tile模板或数据布局的离散模式。

所有示例使用以下导入：

```python
from pypto_pro.runtime.tilingkey import TilingKeyField
import pypto_pro.language as pl
```

---

## 何时使用TilingKey

TilingKey字段在编译阶段会被折叠为常量。因此，每个具体Key都会生成独立Kernel，
不会在单个Kernel中保留对应的运行时分支。TilingKey适合以下场景：

- 可选功能会改变较大代码路径，例如是否应用attention mask；
- Tile尺寸、Layout或Dtype模板只有有限候选值；
- 需要在启动前拒绝不支持的字段组合；
- 二进制交付时需要枚举可编译的TilingKey。

TilingKey不适用于任意运行时shape。候选值的笛卡尔积决定可枚举的组合数量，因此
TilingKey应仅描述有限且会影响代码生成的模式。

---

## 声明Schema

TilingKey是一个普通Python类。类属性使用`TilingKeyField(bits=..., values=...)`声明：

```python
class AttentionKey:
    # 二进制开关，取值只能为 0 或 1。
    HasAtten = TilingKeyField(bits=1, values=[0, 1])

    # 两种固定 Tile 模板。
    BlockM = TilingKeyField(bits=8, values=[64, 128])

    def is_valid(self, key):
        has_atten, block_m = key
        # 仅举例：mask 模式只支持 128 行 Tile。
        return has_atten == 0 or block_m == 128
```

字段按**类定义顺序**收集。该顺序同时决定：

1. `is_valid()`中`key`元组的元素顺序；
2. 64-bit编码中各字段的bit offset；
3. 二进制交付头文件中的字段和selector顺序。

`is_valid()`是可选校验函数，参数`key`使用按照字段定义顺序排列的元组。
该函数既用于在JIT启动时校验具体Key，也用于在二进制交付时过滤枚举组合。

### 字段约束

框架在应用`@pl.jit`装饰器时检查schema：

| 约束 | 说明 |
|---|---|
| `tiling_key` | 必须是class，且至少包含一个`TilingKeyField`。 |
| `bits` | 必须大于0。 |
| `values` | 必须非空、元素必须是互不重复的`int`，不能是`bool`。 |
| 编码容量 | 候选数量不得超过`2**bits`。字段bit保存候选在`values`中的下标，而不是候选值本身。 |
| 总位宽 | 所有字段位宽之和不得超过64。 |
| `is_valid` | 若定义，必须可调用。 |

字段位宽用于为候选**下标**分配编码空间，不限制候选值本身的数值范围。实际值可以是稀疏的
模板编号，例如`bits=3`的候选集合可以是`[16, 64, 128]`；它们分别编码为0、1、2。

### 编码与AscendC对齐

`values`的顺序具有语义：编码后的TilingKey字段保存该值在`values`中的下标，解码后
得到Kernel中使用的实际值。该行为与Ascend C模板TilingKey的Selector一致。

```python
class MaskKey:
    NeedAttnMask = TilingKeyField(bits=1, values=[1, 0])
```

此字段的映射为：

| 实际值 | `values`下标 | 字段Bit / TilingKey |
|---:|---:|---:|
| `1` | `0` | `0` |
| `0` | `1` | `1` |

因此，TilingKey为`1`时，`NeedAttnMask`的实际值为`0`。启动时仍传入实际值，
例如`{"NeedAttnMask": 0}`，不需要手工传入下标。

---

## 将Schema绑定到Kernel

通过`@pl.jit(tiling_key=...)`关联schema。TilingKey字段在Kernel中作为编译期变量直接引用，
无需在Kernel形参列表中声明：

```python
@pl.jit(auto_mutex=True, tiling_key=AttentionKey)
def attention_kernel(
    q: pl.Ptr[pl.DT_FP16],
    k: pl.Ptr[pl.DT_FP16],
    out: pl.Ptr[pl.DT_FP16],
):
    for qi in pl.range(0, 32):
        kv_end = 32
        # HasAtten 在此处是当前 TilingKey 实例对应的编译期常量。
        if HasAtten == 1:
            kv_end = qi + 1
        for ki in pl.range(0, kv_end):
            # ...
            pass
```

例如，`HasAtten=1`时，parser保留`kv_end = qi + 1`分支；`HasAtten=0`时，
编译阶段移除该分支并保留初始值`kv_end = 32`。二者共享源码，但最终Kernel中不会保留对
`HasAtten`的运行时判断。

TilingKey字段名不得与Kernel形参或模块级普通变量冲突。字段名应描述其编译期语义，
例如`HasAtten`、`S1TemplateType`，并避免使用`n`、`shape`等可能与运行时变量冲突的名称。

---

## 选择实例并启动

带TilingKey的Kernel必须在方括号启动参数中提供完整的Key字典，位置在`stream`和
`block_dim`之后：

```python
key = {"HasAtten": 1, "BlockM": 128}
attention_kernel[None, num_cores, key](q, k, out)
```

Key字典的字段必须与Schema **完全一致**：

- 每个字段都必须出现，且不能有额外字段；
- 值必须属于该字段的`values`；
- 整个组合必须通过`is_valid()`；
- 不能直接调用`attention_kernel(...)`，也不能使用list或tuple代替Key字典。

若同时使用`datatype`特化，TilingKey仍是第三个参数，datatype dict紧随其后：

```python
attention_kernel[None, num_cores, key, datatype](q, k, out)
```

框架按照字段定义顺序，将Key字典中的实际值转换为对应的`values`下标，再打包为唯一的
64-bit Key，并缓存对应的专用编译结果。

---

## FlashAttention特化示例

以下示例使用`FaTilingKey`为`flash_attention_score`生成causal attention和
full attention两种专用实例。其他Kernel实参不参与TilingKey Schema，也不影响具体Key的选择。

### `FaTilingKey`的字段

`FaTilingKey`声明14个编译期字段：

| 字段 | bits | 候选值 | 此用例的固定值 |
|---|---:|---|---:|
| `KernelTypeKey` | 2 | 0, 1 | 0 |
| `ImplMode` | 2 | 0, 1, 2 | 0 |
| `Layout` | 4 | 0, 1, 2, 3, 4 | 1 |
| `S1TemplateType` | 10 | 0, 16, 64, 128, 256 | 128 |
| `S2TemplateType` | 10 | 0, 16, 32, 64, 128, 256, 512 | 128 |
| `DTemplateType` | 12 | 0, 16, 32, 48, 64, 80, 96, 128, 160, 192, 256, 768 | 128 |
| `DvTemplateType` | 12 | 同`DTemplateType` | 128 |
| `PseMode` | 4 | 0, 1, 2, 3, 4, 9 | 9 |
| `HasAtten` | 1 | 0, 1 | 0或1 |
| `HasDrop` | 1 | 0, 1 | 0 |
| `HasRope` | 1 | 0, 1 | 0 |
| `OutDtype` | 2 | 0, 1, 2 | 0 |
| `Regbase` | 1 | 0, 1 | 1 |
| `OptionalDn` | 1 | 0, 1 | 0 |

这些字段总计63 bits。`is_valid()`将除`HasAtten`之外的字段限制为表中的固定值，
因此候选值的笛卡尔积最终只保留两个合法Key：`HasAtten=0`和`HasAtten=1`。

Kernel在Cube和Vector两个循环中均使用`HasAtten`选择causal和full专用路径：

```python
causal_skv = skv_tiles
if HasAtten == 1:
    causal_skv = qi + 1
```

`HasAtten=0`的专用Kernel仅保留full attention的`skv_tiles`路径；`HasAtten=1`的
专用Kernel仅保留causal attention的`qi + 1`路径。源码中可以保留清晰的`if/else`结构，
而每个具体Key的最终代码只包含可达分支。

### 启动两个专用实例

基础Key如下：

```python
base_key = {
    "KernelTypeKey": 0, "ImplMode": 0, "Layout": 1,
    "S1TemplateType": 128, "S2TemplateType": 128,
    "DTemplateType": 128, "DvTemplateType": 128,
    "PseMode": 9, "HasAtten": 0, "HasDrop": 0, "HasRope": 0,
    "OutDtype": 0, "Regbase": 1, "OptionalDn": 0,
}

causal_key = {**base_key, "HasAtten": 1}
flash_attention_score[None, actual_num_cores, causal_key, datatype](
    query, key, value, ...
)

full_key = {**base_key, "HasAtten": 0}
flash_attention_score[None, actual_num_cores, full_key, datatype](
    query, key, value, ...
)
```

两次启动使用相同的大部分Key字段，仅改变`HasAtten`，分别选择causal attention和
full attention专用实例。该示例支持FP16和BF16。

---

## 二进制交付

对带TilingKey的Kernel调用`generate_binary_headers()`可生成TilingKey头文件：

```python
from pypto_pro.runtime.opc.pypto_compile import generate_binary_headers

binary_dir = generate_binary_headers(flash_attention_score)
```

生成的`FaTilingKey_tilingkey.h`包含字段声明及通过`is_valid()`
的Key Selector。该文件使用`ASCENDC_TPL_ARGS_DECL`描述各字段和允许值，并以
`ASCENDC_TPL_SEL`仅列出合法组合。字段bit选择`values`中对应下标的实际值；因此应尽量
收紧`values`，并在存在字段关联约束时实现`is_valid()`，避免生成无用的二进制实例。

---

## 常见错误

| 现象 | 原因与处理 |
|---|---|
| 应用Kernel装饰器时失败 | 检查`bits > 0`、候选值为互不重复的整数、候选数量不超过`2**bits`，且总位宽不超过64。 |
| 启动时字段不匹配 | Key字典必须包含所有字段且不能包含未知字段；字段名大小写必须与类属性一致。 |
| 启动值不在候选集中 | 将实际值加入声明的`values`，或使用已有候选值；不能传入候选下标。 |
| 启动被`is_valid()`拒绝 | 按字段定义顺序检查组合约束。 |
| Kernel中找不到字段名 | 在`@pl.jit(tiling_key=...)`中绑定Schema，并避免字段名与Kernel参数或模块变量冲突。 |
| 为每个shape新增Key | 仅将真正影响代码生成的有限模式放入TilingKey。 |

---

## 最小示例

```python
import pypto_pro.language as pl
from pypto_pro.runtime.tilingkey import TilingKeyField


class MyKey:
    UseFastPath = TilingKeyField(bits=1, values=[0, 1])

    def is_valid(self, key):
        (use_fast_path,) = key
        return use_fast_path in (0, 1)


@pl.jit(auto_mutex=True, tiling_key=MyKey)
def kernel(x: pl.Ptr[pl.DT_FP16]):
    if UseFastPath == 1:       # 编译期常量
        pass
    for i in pl.range(0, 8):
        pass


kernel[None, 1, {"UseFastPath": 1}](x)
```

用`TilingKey`选择有限的专用实现，从而消除关键模式分支的运行时开销。
