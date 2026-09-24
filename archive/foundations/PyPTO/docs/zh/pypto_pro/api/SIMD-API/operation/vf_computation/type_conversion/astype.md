# vf.astype

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

`vf.astype`用于数据类型精度转换，将源操作数数据类型转换成目的操作数数据类型，能够实现浮点转整数、浮点转浮点、整数转浮点、整数转整数的数据类型转换。

转换过程中，由于位宽变化、精度变化，支持配置如下参数进行功能实现：

- **layout**（`pl.CastLayout`）：源操作数和目的操作数位宽不同时，单条指令计算量以位宽更大的数据类型为准，`layout`用于控制位宽小的元素在寄存器中的排布方式。`pl.CastLayout.ZERO`放置在偶数索引位置，`pl.CastLayout.ONE`放置在奇数索引位置。FP4类型还支持`pl.CastLayout.TWO`/`pl.CastLayout.THREE`模式（4x扩展/缩窄时使用）。
- **saturate**（`pl.SaturateMode`）：用于设置饱和与不饱和模式。饱和模式下，超出目标类型表示范围的值会被截断为目标类型的最大/最小值；非饱和模式下，超出范围的值行为因转换场景而异（详见[约束说明](#约束说明)）。
- **mode**（`pl.MergeMode`）：用于指定写入寄存器数据模式，`preg`未选择的元素在dst中置零（`pl.MergeMode.ZEROING`）或保留dst原值（`pl.MergeMode.MERGING`）。当前设备仅支持`pl.MergeMode.ZEROING`模式。
- **round_mode**（`pl.VFRoundMode`）：用于设置舍入模式。仅在可能导致精度损失且支持该舍入模式的转换中生效。

不同数据类型下元素对应的`preg`位宽不一致，在类型转换时，mask_tensor根据输入的源操作数进行有效元素筛选。当源操作数和目的操作数位宽不同时，单条指令计算量以位宽更大的数据类型为准，layout用于控制位宽小的元素在寄存器中的排布方式。

下图展示了mask_tensor和layout同时作用时16位宽和32位宽进行类型转换的过程：

**图1** 16位宽类型到32位宽类型转换过程

![](../../../../figures/astype_b16_to_b32_conversion.jpg)

**图2** 32位宽类型到16位宽类型转换过程

![](../../../../figures/astype_b32_to_b16_conversion.jpg)

特别地，DT_FP4E2M1、DT_FP4E1M2与DT_BF16之间的转换，指令会以每2个元素为一对进行读写，大转小时`preg`有效位以偶数位为准。下图展示了mask_tensor和layout同时作用时DT_FP4E2M1和DT_BF16之间的转换过程：

**图3** DT_FP4E2M1到DT_BF16类型转换过程

![](../../../../figures/astype_fp4x2_e2m1_to_bf16_conversion.jpg)

**图4** DT_BF16到DT_FP4E2M1类型转换过程

![](../../../../figures/astype_bf16_to_fp4x2_e2m1_conversion.jpg)

## 函数原型

```python
astype(src, preg, dtype: Optional[DType] = None, layout: Optional[CastLayout] = None, round_mode: Optional[VFRoundMode] = None, saturate: Optional[SaturateMode] = None)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `src` | 输入 | 源操作数，reg_tensor，支持的数据类型请参见[约束说明](#约束说明)。 |
| `preg` | 输入 | mask_tensor。`preg`会按照输入的源操作数来筛选有效元素。 |
| `dtype` | 输入 | 必选，指定目标寄存器的数据类型（如`pl.DT_FP16`、`pl.DT_INT32`等）。由于类型转换后目标类型与源类型不同，必须显式指定。 |
| `layout` | 输入 | 可选，[CastLayout](../types/CastLayout.md)枚举类型。`pl.CastLayout.ZERO`（偶数半区，默认）或`pl.CastLayout.ONE`（奇数半区）。当源操作数和目的操作数位宽不同时，控制位宽小的元素在寄存器中的排布方式。FP4类型还支持`pl.CastLayout.TWO`/`pl.CastLayout.THREE`模式。具体支持的值因转换路径而异，详见[约束说明](#约束说明)各表。 |
| `round_mode` | 输入 | 可选，[VFRoundMode](../types/VFRoundMode.md)枚举类型，浮点舍入模式。默认`pl.VFRoundMode.CAST_RINT`。不同转换路径支持的舍入模式不同，详见[约束说明](#约束说明)各表。不涉及精度损失的转换路径round_mode标记为`UNKNOWN`（可省略）。 |
| `saturate` | 输入 | 可选，[SaturateMode](../types/SaturateMode.md)枚举类型。`pl.SaturateMode.OFF`（默认，非饱和）或`pl.SaturateMode.ON`（饱和）。具体支持的值因转换路径而异，详见[约束说明](#约束说明)各表。标记为`UNKNOWN`的路径表示不涉及饱和/非饱和选择。 |
| `mode` | 输入 | 可选，对应[MergeMode](../types/MergeMode.md)类型。<br>- `pl.MergeMode.ZEROING`（默认），`preg`未筛选的元素在`dst`中置0。<br>- `pl.MergeMode.MERGING`当前不支持。 |

## 约束说明

- 数据类型约束：

  **表1** 支持的数据类型转换

  | src | dst |
  |---|---|
  | DT_INT4 | DT_INT16、DT_FP16、DT_BF16 |
  | DT_INT8 | DT_INT16、DT_FP16、DT_INT32 |
  | DT_UINT8 | DT_UINT16、DT_FP16、DT_UINT32 |
  | DT_FP4E2M1 | DT_BF16 |
  | DT_FP4E1M2 | DT_BF16 |
  | DT_HF8 | DT_FP16、DT_FP32 |
  | DT_FP8E8M0 | DT_BF16 |
  | DT_FP8E5M2 | DT_FP32 |
  | DT_FP8E4M3FN | DT_FP32 |
  | DT_INT16 | DT_INT4、DT_UINT8、DT_FP16、DT_INT32、DT_UINT32、DT_FP32 |
  | DT_UINT16 | DT_UINT8、DT_UINT32 |
  | DT_FP16 | DT_INT4、DT_INT8、DT_UINT8、DT_HF8、DT_INT16、DT_BF16、DT_INT32、DT_FP32 |
  | DT_BF16 | DT_FP4E2M1、DT_FP4E1M2、DT_FP8E8M0、DT_FP16、DT_INT32、DT_FP32 |
  | DT_INT32 | DT_UINT8、DT_INT16、DT_UINT16、DT_FP32、DT_INT64 |
  | DT_UINT32 | DT_UINT8、DT_INT16、DT_UINT16 |
  | DT_FP32 | DT_HF8、DT_FP8E5M2、DT_FP8E4M3FN、DT_INT16、DT_FP16、DT_BF16、DT_INT32、DT_INT64 |
  | DT_INT64 | DT_INT32、DT_FP32 |

- 不同场景下的参数类型转换：

  **表2** 浮点转整数

  | src | dst | layout | saturate | mode | round_mode |
  |---|---|---|---|---|---|
  | DT_FP16 | DT_INT4 | pl.CastLayout.ZERO/pl.CastLayout.ONE/pl.CastLayout.TWO/pl.CastLayout.THREE | pl.SaturateMode.OFF/pl.SaturateMode.ON | pl.MergeMode.ZEROING | pl.VFRoundMode.CAST_RINT/pl.VFRoundMode.CAST_ROUND/pl.VFRoundMode.CAST_FLOOR/pl.VFRoundMode.CAST_CEIL/pl.VFRoundMode.CAST_TRUNC |
  | DT_FP16 | DT_INT8 | pl.CastLayout.ZERO/pl.CastLayout.ONE | pl.SaturateMode.OFF/pl.SaturateMode.ON | pl.MergeMode.ZEROING | pl.VFRoundMode.CAST_RINT/pl.VFRoundMode.CAST_ROUND/pl.VFRoundMode.CAST_FLOOR/pl.VFRoundMode.CAST_CEIL/pl.VFRoundMode.CAST_TRUNC |
  | DT_FP16 | DT_UINT8 | pl.CastLayout.ZERO/pl.CastLayout.ONE | pl.SaturateMode.OFF/pl.SaturateMode.ON | pl.MergeMode.ZEROING | pl.VFRoundMode.CAST_RINT/pl.VFRoundMode.CAST_ROUND/pl.VFRoundMode.CAST_FLOOR/pl.VFRoundMode.CAST_CEIL/pl.VFRoundMode.CAST_TRUNC |
  | DT_FP16 | DT_INT16 | UNKNOWN | pl.SaturateMode.OFF/pl.SaturateMode.ON | pl.MergeMode.ZEROING | pl.VFRoundMode.CAST_RINT/pl.VFRoundMode.CAST_ROUND/pl.VFRoundMode.CAST_FLOOR/pl.VFRoundMode.CAST_CEIL/pl.VFRoundMode.CAST_TRUNC |
  | DT_FP16 | DT_INT32 | pl.CastLayout.ZERO/pl.CastLayout.ONE | UNKNOWN | pl.MergeMode.ZEROING | pl.VFRoundMode.CAST_RINT/pl.VFRoundMode.CAST_ROUND/pl.VFRoundMode.CAST_FLOOR/pl.VFRoundMode.CAST_CEIL/pl.VFRoundMode.CAST_TRUNC |
  | DT_BF16 | DT_INT32 | pl.CastLayout.ZERO/pl.CastLayout.ONE | pl.SaturateMode.OFF/pl.SaturateMode.ON | pl.MergeMode.ZEROING | pl.VFRoundMode.CAST_RINT/pl.VFRoundMode.CAST_ROUND/pl.VFRoundMode.CAST_FLOOR/pl.VFRoundMode.CAST_CEIL/pl.VFRoundMode.CAST_TRUNC |
  | DT_FP32 | DT_INT16 | pl.CastLayout.ZERO/pl.CastLayout.ONE | pl.SaturateMode.OFF/pl.SaturateMode.ON | pl.MergeMode.ZEROING | pl.VFRoundMode.CAST_RINT/pl.VFRoundMode.CAST_ROUND/pl.VFRoundMode.CAST_FLOOR/pl.VFRoundMode.CAST_CEIL/pl.VFRoundMode.CAST_TRUNC |
  | DT_FP32 | DT_INT32 | UNKNOWN | pl.SaturateMode.OFF/pl.SaturateMode.ON | pl.MergeMode.ZEROING | pl.VFRoundMode.CAST_RINT/pl.VFRoundMode.CAST_ROUND/pl.VFRoundMode.CAST_FLOOR/pl.VFRoundMode.CAST_CEIL/pl.VFRoundMode.CAST_TRUNC |
  | DT_FP32 | DT_INT64 | pl.CastLayout.ZERO/pl.CastLayout.ONE | pl.SaturateMode.OFF/pl.SaturateMode.ON | pl.MergeMode.ZEROING | pl.VFRoundMode.CAST_RINT/pl.VFRoundMode.CAST_ROUND/pl.VFRoundMode.CAST_FLOOR/pl.VFRoundMode.CAST_CEIL/pl.VFRoundMode.CAST_TRUNC |

  **表3** 浮点转浮点

  | src | dst | layout | saturate | mode | round_mode |
  |---|---|---|---|---|---|
  | DT_HF8 | DT_FP16 | pl.CastLayout.ZERO/pl.CastLayout.ONE | UNKNOWN | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_HF8 | DT_FP32 | pl.CastLayout.ZERO/pl.CastLayout.ONE/pl.CastLayout.TWO/pl.CastLayout.THREE | UNKNOWN | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_FP8E4M3FN | DT_FP32 | pl.CastLayout.ZERO/pl.CastLayout.ONE/pl.CastLayout.TWO/pl.CastLayout.THREE | UNKNOWN | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_FP8E5M2 | DT_FP32 | pl.CastLayout.ZERO/pl.CastLayout.ONE/pl.CastLayout.TWO/pl.CastLayout.THREE | UNKNOWN | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_FP8E8M0 | DT_BF16 | pl.CastLayout.ZERO/pl.CastLayout.ONE | UNKNOWN | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_FP4E2M1 | DT_BF16 | pl.CastLayout.ZERO/pl.CastLayout.ONE/pl.CastLayout.TWO/pl.CastLayout.THREE | UNKNOWN | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_FP4E1M2 | DT_BF16 | pl.CastLayout.ZERO/pl.CastLayout.ONE/pl.CastLayout.TWO/pl.CastLayout.THREE | UNKNOWN | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_FP16 | DT_HF8 | pl.CastLayout.ZERO/pl.CastLayout.ONE | pl.SaturateMode.OFF/pl.SaturateMode.ON | pl.MergeMode.ZEROING | pl.VFRoundMode.CAST_ROUND/pl.VFRoundMode.CAST_HYBRID |
  | DT_FP16 | DT_BF16 | UNKNOWN | UNKNOWN | pl.MergeMode.ZEROING | pl.VFRoundMode.CAST_RINT/pl.VFRoundMode.CAST_ROUND/pl.VFRoundMode.CAST_FLOOR/pl.VFRoundMode.CAST_CEIL/pl.VFRoundMode.CAST_TRUNC |
  | DT_BF16 | DT_FP16 | UNKNOWN | pl.SaturateMode.OFF/pl.SaturateMode.ON | pl.MergeMode.ZEROING | pl.VFRoundMode.CAST_RINT/pl.VFRoundMode.CAST_ROUND/pl.VFRoundMode.CAST_FLOOR/pl.VFRoundMode.CAST_CEIL/pl.VFRoundMode.CAST_TRUNC |
  | DT_BF16 | DT_FP4E2M1 | pl.CastLayout.ZERO/pl.CastLayout.ONE/pl.CastLayout.TWO/pl.CastLayout.THREE | UNKNOWN | pl.MergeMode.ZEROING | pl.VFRoundMode.CAST_RINT/pl.VFRoundMode.CAST_ROUND/pl.VFRoundMode.CAST_FLOOR/pl.VFRoundMode.CAST_CEIL/pl.VFRoundMode.CAST_TRUNC |
  | DT_BF16 | DT_FP4E1M2 | pl.CastLayout.ZERO/pl.CastLayout.ONE/pl.CastLayout.TWO/pl.CastLayout.THREE | UNKNOWN | pl.MergeMode.ZEROING | pl.VFRoundMode.CAST_RINT/pl.VFRoundMode.CAST_ROUND/pl.VFRoundMode.CAST_FLOOR/pl.VFRoundMode.CAST_CEIL/pl.VFRoundMode.CAST_TRUNC |
  | DT_BF16 | DT_FP8E8M0 | pl.CastLayout.ZERO/pl.CastLayout.ONE | UNKNOWN | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_BF16 | DT_FP32 | pl.CastLayout.ZERO/pl.CastLayout.ONE | UNKNOWN | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_FP32 | DT_HF8 | pl.CastLayout.ZERO/pl.CastLayout.ONE/pl.CastLayout.TWO/pl.CastLayout.THREE | pl.SaturateMode.OFF/pl.SaturateMode.ON | pl.MergeMode.ZEROING | pl.VFRoundMode.CAST_ROUND/pl.VFRoundMode.CAST_HYBRID |
  | DT_FP32 | DT_FP8E4M3FN | pl.CastLayout.ZERO/pl.CastLayout.ONE/pl.CastLayout.TWO/pl.CastLayout.THREE | pl.SaturateMode.OFF/pl.SaturateMode.ON | pl.MergeMode.ZEROING | pl.VFRoundMode.CAST_RINT |
  | DT_FP32 | DT_FP8E5M2 | pl.CastLayout.ZERO/pl.CastLayout.ONE/pl.CastLayout.TWO/pl.CastLayout.THREE | pl.SaturateMode.OFF/pl.SaturateMode.ON | pl.MergeMode.ZEROING | pl.VFRoundMode.CAST_RINT |
  | DT_FP32 | DT_FP16 | pl.CastLayout.ZERO/pl.CastLayout.ONE | pl.SaturateMode.OFF/pl.SaturateMode.ON | pl.MergeMode.ZEROING | pl.VFRoundMode.CAST_ODD/pl.VFRoundMode.CAST_RINT/pl.VFRoundMode.CAST_ROUND/pl.VFRoundMode.CAST_FLOOR/pl.VFRoundMode.CAST_CEIL/pl.VFRoundMode.CAST_TRUNC |
  | DT_FP32 | DT_BF16 | pl.CastLayout.ZERO/pl.CastLayout.ONE | pl.SaturateMode.OFF/pl.SaturateMode.ON | pl.MergeMode.ZEROING | pl.VFRoundMode.CAST_RINT/pl.VFRoundMode.CAST_ROUND/pl.VFRoundMode.CAST_FLOOR/pl.VFRoundMode.CAST_CEIL/pl.VFRoundMode.CAST_TRUNC |

  **表4** 整数转浮点

  | src | dst | layout | saturate | mode | round_mode |
  |---|---|---|---|---|---|
  | DT_INT4 | DT_FP16 | pl.CastLayout.ZERO/pl.CastLayout.ONE/pl.CastLayout.TWO/pl.CastLayout.THREE | UNKNOWN | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_INT4 | DT_BF16 | pl.CastLayout.ZERO/pl.CastLayout.ONE/pl.CastLayout.TWO/pl.CastLayout.THREE | UNKNOWN | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_INT8 | DT_FP16 | pl.CastLayout.ZERO/pl.CastLayout.ONE | UNKNOWN | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_UINT8 | DT_FP16 | pl.CastLayout.ZERO/pl.CastLayout.ONE | UNKNOWN | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_INT16 | DT_FP16 | UNKNOWN | UNKNOWN | pl.MergeMode.ZEROING | pl.VFRoundMode.CAST_RINT/pl.VFRoundMode.CAST_ROUND/pl.VFRoundMode.CAST_FLOOR/pl.VFRoundMode.CAST_CEIL/pl.VFRoundMode.CAST_TRUNC |
  | DT_INT16 | DT_FP32 | pl.CastLayout.ZERO/pl.CastLayout.ONE | UNKNOWN | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_INT32 | DT_FP32 | UNKNOWN | UNKNOWN | pl.MergeMode.ZEROING | pl.VFRoundMode.CAST_RINT/pl.VFRoundMode.CAST_ROUND/pl.VFRoundMode.CAST_FLOOR/pl.VFRoundMode.CAST_CEIL/pl.VFRoundMode.CAST_TRUNC |
  | DT_INT64 | DT_FP32 | pl.CastLayout.ZERO/pl.CastLayout.ONE | UNKNOWN | pl.MergeMode.ZEROING | pl.VFRoundMode.CAST_RINT/pl.VFRoundMode.CAST_ROUND/pl.VFRoundMode.CAST_FLOOR/pl.VFRoundMode.CAST_CEIL/pl.VFRoundMode.CAST_TRUNC |

  **表5** 整数转整数

  | src | dst | layout | saturate | mode | round_mode |
  |---|---|---|---|---|---|
  | DT_INT4 | DT_INT16 | pl.CastLayout.ZERO/pl.CastLayout.ONE/pl.CastLayout.TWO/pl.CastLayout.THREE | UNKNOWN | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_INT8 | DT_INT16 | pl.CastLayout.ZERO/pl.CastLayout.ONE | UNKNOWN | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_INT8 | DT_INT32 | pl.CastLayout.ZERO/pl.CastLayout.ONE/pl.CastLayout.TWO/pl.CastLayout.THREE | UNKNOWN | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_UINT8 | DT_UINT16 | pl.CastLayout.ZERO/pl.CastLayout.ONE | UNKNOWN | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_UINT8 | DT_UINT32 | pl.CastLayout.ZERO/pl.CastLayout.ONE/pl.CastLayout.TWO/pl.CastLayout.THREE | UNKNOWN | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_INT16 | DT_INT4 | pl.CastLayout.ZERO/pl.CastLayout.ONE/pl.CastLayout.TWO/pl.CastLayout.THREE | pl.SaturateMode.OFF/pl.SaturateMode.ON | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_INT16 | DT_UINT8 | pl.CastLayout.ZERO/pl.CastLayout.ONE | pl.SaturateMode.OFF/pl.SaturateMode.ON | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_INT16 | DT_INT32 | pl.CastLayout.ZERO/pl.CastLayout.ONE | UNKNOWN | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_INT16 | DT_UINT32 | pl.CastLayout.ZERO/pl.CastLayout.ONE | UNKNOWN | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_UINT16 | DT_UINT8 | pl.CastLayout.ZERO/pl.CastLayout.ONE | pl.SaturateMode.OFF/pl.SaturateMode.ON | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_UINT16 | DT_UINT32 | pl.CastLayout.ZERO/pl.CastLayout.ONE | UNKNOWN | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_INT32 | DT_UINT8 | pl.CastLayout.ZERO/pl.CastLayout.ONE/pl.CastLayout.TWO/pl.CastLayout.THREE | pl.SaturateMode.OFF/pl.SaturateMode.ON | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_INT32 | DT_INT16 | pl.CastLayout.ZERO/pl.CastLayout.ONE | pl.SaturateMode.OFF/pl.SaturateMode.ON | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_INT32 | DT_UINT16 | pl.CastLayout.ZERO/pl.CastLayout.ONE | pl.SaturateMode.OFF/pl.SaturateMode.ON | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_INT32 | DT_INT64 | pl.CastLayout.ZERO/pl.CastLayout.ONE | UNKNOWN | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_UINT32 | DT_UINT8 | pl.CastLayout.ZERO/pl.CastLayout.ONE/pl.CastLayout.TWO/pl.CastLayout.THREE | pl.SaturateMode.OFF/pl.SaturateMode.ON | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_UINT32 | DT_INT16 | pl.CastLayout.ZERO/pl.CastLayout.ONE | pl.SaturateMode.OFF/pl.SaturateMode.ON | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_UINT32 | DT_UINT16 | pl.CastLayout.ZERO/pl.CastLayout.ONE | pl.SaturateMode.OFF/pl.SaturateMode.ON | pl.MergeMode.ZEROING | UNKNOWN |
  | DT_INT64 | DT_INT32 | pl.CastLayout.ZERO/pl.CastLayout.ONE | pl.SaturateMode.OFF/pl.SaturateMode.ON | pl.MergeMode.ZEROING | UNKNOWN |

  > **表中`UNKNOWN`的含义**：`layout`为`UNKNOWN`表示该转换路径源和目的位宽相同，无需指定layout（可省略）。`saturate`为`UNKNOWN`表示该转换路径不涉及饱和/非饱和选择（可省略）。`round_mode`为`UNKNOWN`表示该转换路径不涉及精度损失（可省略）。

- 饱和和不饱和模式说明：

  **表6** 不同类型转换场景下的不饱和模式和饱和模式

  | 场景 | 不饱和模式 | 饱和模式 |
  |---|---|---|
  | 浮点转整数 | 输入数据超过输出类型最值时，结果被截断为目标格式的数据宽度（保留最低有效位），例如输入half值为257，输出uint8_t值为1；输入为+/-inf时，则返回输出类型的对应最值；输入为nan时，返回0。 | 输入数据超过输出类型最值时，返回输出类型的对应最值，例如输入half值为257，输出uint8值为255，输入half值为-inf，输出uint8_t值为0；输入为nan时，返回0。 |
  | 浮点转浮点 | 输入数据为nan时，输出为nan；输入+/-inf时，输出为+/-inf。 | 输入为nan时，输出为0；输入数据超过输出类型最值时，返回输出类型的对应最值。 |
  | 整数转浮点 | 不支持不饱和模式 | 输入为nan时，输出为0；输入数据超过输出类型最值时，返回输出类型的对应最值。该场景默认饱和模式，无需配置。 |
  | 整数转整数 | 输入数据会截断为目标数据宽度，例如，输入int32_t值为256，输出uint8_t值为0。 | 输入数据超出目标数据范围，会饱和为目标数据最值。 |

- 浮点转浮点的特殊约束：

    - 当输出类型为FP32时，只支持不饱和模式。
    - 不饱和模式：当输出类型为FP8E4M3FN时，由于FP8E4M3FN没有inf表示格式，所以输出为nan。
    - 饱和模式：当输出类型为FP8E5M2/FP8E4M3FN时，输入nan默认输出为0。如果CTRL[50] = 1'b1，则输出为nan。
    - FP4E2M1/FP4E1M2数据类型没有inf和nan的定义。对于BF16到FP4的转换，输入BF16类型的值为inf或超出FP4数据最值范围时，会返回对应符号的FP4最值；输入nan时，FP4输出0。
    - 对于FP8E8M0类型：输入BF16 +/-inf或绝对值超出FP8E8M0类型最大值，则返回FP8E8M0最大值0b11111110；输入BF16 nan输出FP8E8M0 nan = 0b11111111。

- 整数转整数的特殊约束

    - 对于窄数据类型例如INT16(2Byte)转宽数据类型UINT32(4Byte)，只支持饱和模式，输入负数会被饱和成0。

## 返回值说明

返回`dst`目的操作数，reg_tensor，支持的数据类型请参见[约束说明](#约束说明)。当目的操作数位宽比源操作数小时，在`preg`和`layout`作用下，目的操作数中的无效元素均为0

## 调用示例

### BF16转FP32

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_tile, 0)
    # FP32→BF16，layout=pl.CastLayout.ZERO 放偶数半区
    reg_bf16 = vf.astype(reg_a, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO)
    # BF16→FP32，widen back for store
    reg_f32 = vf.astype(reg_bf16, preg, dtype=pl.DT_FP32)
    vf.store_align(dst_tile, reg_f32, preg)


@pl.jit()
def example_kernel(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf(in_a, t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])


def test_example():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a.to(torch.bfloat16).to(torch.float32), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_example()
    print("PASSED")
```

### FP32转FP16

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf_fp32_to_fp16(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_tile, 0)
    # FP32→FP16，缩窄转换，layout=pl.CastLayout.ZERO 放偶数半区
    reg_f16 = vf.astype(reg_a, preg, dtype=pl.DT_FP16, layout=pl.CastLayout.ZERO)
    # FP16→FP32，扩展回 FP32 用于搬出
    reg_f32 = vf.astype(reg_f16, preg, dtype=pl.DT_FP32)
    vf.store_align(dst_tile, reg_f32, preg)


@pl.jit()
def example_kernel_fp32_to_fp16(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf_fp32_to_fp16(in_a, t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])


def test_example_fp32_to_fp16():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel_fp32_to_fp16[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a.to(torch.float16).to(torch.float32), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_example_fp32_to_fp16()
    print("PASSED")
```

### FP32转INT32

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf_round(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_tile, 0)
    # FP32→INT32，round_mode=CAST_ODD 使用 Von Neumann 舍入
    reg_i = vf.astype(reg_a, preg, dtype=pl.DT_INT32, round_mode=pl.VFRoundMode.CAST_RINT)
    reg_f = vf.astype(reg_i, preg, dtype=pl.DT_FP32)
    vf.store_align(dst_tile, reg_f, preg)


@pl.jit()
def example_kernel_round(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf_round(in_a, t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])


def test_example_2():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel_round[None, core_nums](a, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a.to(torch.int32).to(torch.float32), rtol=0, atol=1.0)


if __name__ == "__main__":
    test_example_2()
    print("PASSED")
```

### FP32转FP8E4M3FN

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf_fp8(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    reg_a = vf.load_align(src_tile, 0)
    # FP32 → FP8E4M3FN，layout=ZERO 放偶数半区，仅支持 CAST_RINT
    reg_f8 = vf.astype(reg_a, preg, dtype=pl.DT_FP8E4M3FN, layout=pl.CastLayout.ZERO,
                       round_mode=pl.VFRoundMode.CAST_RINT, saturate=pl.SaturateMode.ON)
    # FP8E4M3FN → FP32，还原为 FP32 用于存储
    reg_f32 = vf.astype(reg_f8, preg, dtype=pl.DT_FP32)
    vf.store_align(dst_tile, reg_f32, preg)


@pl.jit()
def example_kernel_fp8(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
):
    tf = pl.TileType(shape=[1, 64], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf_fp8(in_a, t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])


def test_example_fp8():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 64], device=device, dtype=torch.float32)
    out = torch.empty([1, 64], device=device, dtype=torch.float32)
    example_kernel_fp8[None, core_nums](a, out)
    torch.npu.synchronize()
    # layout=ZERO 时 FP8 结果在偶数半区，验证偶数索引位置
    expected = a.to(torch.float8_e4m3fn).to(torch.float32)
    torch.testing.assert_close(out[:, ::2], expected[:, ::2], rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test_example_fp8()
    print("PASSED")
```

### BF16转FP4E2M1

```python
import numpy as np
import os
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf_fp4(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    reg_a = vf.load_align(src_tile, 0, dtype=pl.DT_BF16)
    # BF16 → FP4E2M1，layout=ZERO 放偶数半区
    reg_f4 = vf.astype(reg_a, preg, dtype=pl.DT_FP4E2M1, layout=pl.CastLayout.ZERO,
                       round_mode=pl.VFRoundMode.CAST_RINT)
    # FP4E2M1 → BF16，还原为 BF16 用于存储
    reg_bf16 = vf.astype(reg_f4, preg, dtype=pl.DT_BF16)
    vf.store_align(dst_tile, reg_bf16, preg)


@pl.jit()
def example_kernel_fp4(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_BF16],
):
    tf = pl.TileType(shape=[1, 128], dtype=pl.DT_BF16, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf_fp4(in_a, t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])


# FP4 E2M1 可表示值为 [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] 及其负数
# 硬件采用 ROUND_R（round-to-nearest-even）舍入
_FP4_E2M1_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
_FP4_E2M1_EVEN_MASK = np.array([True, False, True, False, True, False, True, False])


def test_example_fp4():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 128], device=device, dtype=torch.bfloat16)
    out = torch.empty([1, 128], device=device, dtype=torch.bfloat16)
    example_kernel_fp4[None, core_nums](a, out)
    torch.npu.synchronize()
    # layout=ZERO 时 FP4 结果在偶数半区，验证偶数索引位置
    a_np = a.cpu().float().numpy()
    sign = np.sign(a_np)
    abs_val = np.abs(a_np)
    # 计算到每个可表示值的距离
    dist = np.abs(abs_val[..., None] - _FP4_E2M1_VALUES)
    min_dist = dist.min(axis=-1)
    # 找到所有距离最小的候选值
    is_nearest = dist == min_dist[..., None]
    # 当存在多个等距候选时，选择 code 为偶数的值（round-to-nearest-even）
    # FP4 E2M1 的 code 为 [0,1,2,3,4,5,6,7]，偶数 code 对应索引 0,2,4,6
    candidates = is_nearest & _FP4_E2M1_EVEN_MASK
    # 若有偶数候选则选它，否则选唯一的最近值
    has_even = candidates.any(axis=-1)
    idx = np.where(has_even, np.argmax(candidates, axis=-1), np.argmin(dist, axis=-1))
    expected = torch.from_numpy(sign * _FP4_E2M1_VALUES[idx]).to(device=device, dtype=torch.bfloat16)
    torch.testing.assert_close(out[:, ::2], expected[:, ::2], rtol=1e-1, atol=1e-1)


if __name__ == "__main__":
    test_example_fp4()
    print("PASSED")
```

### FP16转HF8

```python
import os
import pypto_pro.language as pl
import torch
import torch_npu


@pl.vector_function
def example_vf_hf8(src_tile, dst_tile):
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    reg_a = vf.load_align(src_tile, 0, dtype=pl.DT_FP16)
    # FP16 → HF8（2x narrowing），layout=ZERO 放偶数半区
    reg_hf8 = vf.astype(reg_a, preg, dtype=pl.DT_HF8, layout=pl.CastLayout.ZERO,
                        round_mode=pl.VFRoundMode.CAST_ROUND, saturate=pl.SaturateMode.ON)
    # HF8 → FP16（2x widening），还原为 FP16 用于存储
    reg_f16 = vf.astype(reg_hf8, preg, dtype=pl.DT_FP16)
    vf.store_align(dst_tile, reg_f16, preg)


@pl.jit()
def example_kernel_hf8(
    a: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
):
    tf = pl.TileType(shape=[1, 128], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    in_a_grp = pl.make_tile_group(type=tf, addrs=0x0, mutex_ids=[0])
    in_a = in_a_grp.current()
    t_out_grp = pl.make_tile_group(type=tf, addrs=0x100, mutex_ids=[1])
    t_out = t_out_grp.current()
    with pl.section_vector():
        pl.load(in_a, a, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        example_vf_hf8(in_a, t_out)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.store(out, t_out, [0, 0])


def test_example_hf8():
    device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", 0))
    device = f"npu:{device_id}"
    core_nums = 1
    torch.npu.set_device(device)
    a = torch.randn([1, 128], device=device, dtype=torch.float16)
    out = torch.empty([1, 128], device=device, dtype=torch.float16)
    example_kernel_hf8[None, core_nums](a, out)
    torch.npu.synchronize()
    # layout=ZERO 时 HF8 结果在偶数半区，验证偶数索引位置
    torch.testing.assert_close(out[:, ::2], a[:, ::2], rtol=1e-1, atol=1e-1)


if __name__ == "__main__":
    test_example_hf8()
    print("PASSED")
```
