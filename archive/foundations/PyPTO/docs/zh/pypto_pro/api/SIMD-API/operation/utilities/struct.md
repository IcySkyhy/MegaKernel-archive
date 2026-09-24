# pypto_pro.language.struct

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

编译期结构体，将多个标量值组合为一个具名字段的结构体变量。

PyPTO Kernel最终编译为NPU上的C++ 代码，`pl.struct`在编译期生成对应的C++ struct定义和实例。典型用途是通过[`ssbuf_store`](../memory_data_movement/ssbuf_store.md)/[`ssbuf_load`](../memory_data_movement/ssbuf_load.md)在SSBUF（共享标量缓冲区）中传递少量元数据（如批次号、块号、地址偏移），实现不同流水线（pipe）或不同核之间的标量通信。

> **关于pipe**：NPU内部有多条并行流水线（MTE1/MTE2/MTE3/M/V/S/FIX），各自负责不同阶段的数据搬运和计算。详见[`PipeType`](../../basic_data_structures/PipeType.md)。

## 函数原型

```python
pypto_pro.language.struct("TypeName", field1=default1, field2=default2, ...)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `"TypeName"` | 输入 | 结构体类型名（字符串常量） |
| `field=value` | 输入 | 字段名和初始值（关键字参数） |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `"TypeName"` | 输入 | 必须是字符串常量，不能是变量<br>须为合法的C标识符（字母/数字/下划线，不以数字开头）<br>编译后生成的C++ struct以此命名 |
| `field=value` | 输入 | 至少一个关键字参数<br>字段名须为合法的C标识符，不可重复<br>初始值可为整型、浮点型、标量表达式或列表（数组字段，如`arr=[0, 0, 0, 0]`）<br>不支持`**kwargs`展开<br>字段值可多次修改（通过`message.field = new_value`赋值）<br>数组字段元素可通过`s.arr_field[idx] = val`赋值，通过`s.arr_field[idx]`读取<br>可在`for`循环内创建struct；`if/else`分支内只能修改字段，不能创建新struct（须在分支外创建）<br>字段数量已验证8个，SSBUF容量有限（通常256字节），传递的struct不宜过大 |

## 调用示例
### 示例一

下面是一个完整Kernel：Vector侧把struct（含标量和数组字段）写入SSBUF并发送跨核事件，Cube侧等待后读取，再写回输出Tensor。

```python
import pypto_pro.language as pl


@pl.jit()
def struct_kernel(out: pl.Tensor[[6], pl.DT_INT32]):
    # 创建结构体：标量字段 + 数组字段
    message = pl.struct("Message", batch=0, block=0, offsets=[0, 0, 0, 0])

    # 修改标量字段
    message.batch = 8
    message.block = 1

    # 修改数组字段元素
    message.offsets[0] = 32768
    message.offsets[2] = 65536

    # Vector 侧写入 SSBUF，Cube 侧读取
    with pl.section_vector():
        if pl.get_subblock_idx() == 0:
            pl.ssbuf_store(message, 0)
            pl.system.set_cross_core(pipe=pl.PipeType.S, event_id=15)

    with pl.section_cube():
        pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=15, sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
        pl.ssbuf_load(message, 0)
        pl.printf("batch=%d, block=%d, offset0=%d",
                  message.batch, message.block, message.offsets[0])

    # 读回数组字段元素并输出
    with pl.section_vector():
        pl.setval(out, 0, message.batch)
        pl.setval(out, 1, message.block)
        pl.setval(out, 2, message.offsets[0])
        pl.setval(out, 3, message.offsets[2])
        pl.setval(out, 4, message.offsets[1])
        pl.setval(out, 5, message.offsets[3])
```

### 示例二

下面是一个完整Kernel：for循环中按索引读写数组字段元素，并做累加。

```python
import pypto_pro.language as pl


@pl.jit()
def struct_array_field_kernel(out: pl.Tensor[[5], pl.DT_INT32]):
    # 创建带数组字段的结构体
    s = pl.struct("RunInfo", batch_id=0, offsets=[0, 0, 0, 0])

    with pl.section_vector():
        # 数组字段元素赋值（s.arr_field[idx] = val）
        s.offsets[0] = 10
        s.offsets[1] = 20
        s.offsets[2] = 30
        s.offsets[3] = 40

        # 数组字段元素读取（s.arr_field[idx]）
        total = 0
        for i in pl.range(0, 4):
            total = total + s.offsets[i]
        pl.setval(out, 0, s.offsets[0])
        pl.setval(out, 1, s.offsets[3])
        pl.setval(out, 2, total)
        pl.setval(out, 3, s.batch_id)
        pl.setval(out, 4, s.offsets[1] + s.offsets[2])
```
