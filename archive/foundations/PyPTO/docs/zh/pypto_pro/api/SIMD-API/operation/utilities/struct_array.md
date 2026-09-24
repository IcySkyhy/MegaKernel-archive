# pypto_pro.language.struct_array

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

结构体数组，N个相同的struct按索引存取。用于流水线 / FIFO场景中按槽位索引存取上下文。

## 函数原型

```python
pypto_pro.language.struct_array(size, "TypeName", field1=default1, ...)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `size` | 输入 | 数组长度（正整数） |
| `"TypeName"` | 输入 | 结构体类型名（字符串常量） |
| `field=value` | 输入 | 字段名和初始值（关键字参数） |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `size` | 输入 | 必须是编译时常量正整数（`size >= 1`）<br>非常量或非正整数报`ParserSyntaxError` |
| `"TypeName"` | 输入 | 必须是字符串常量，作为第二个位置参数<br>缺失或非字符串报`ParserSyntaxError` |
| `field=value` | 输入 | 至少一个关键字参数<br>字段名须为合法标识符<br>初始值可为整型、浮点型、标量表达式或列表（数组字段，如`arr=[0, 0, 0, 0]`）<br>不支持`**kwargs`展开<br>数组字段元素可通过`arr[i].field[j] = val`赋值，通过`arr[i].field[j]`读取 |
| `index`（存取时） | 输入 | 整数常量或整数标量表达式（如`task_id % size`、循环变量`i`）<br>不支持负数索引、切片、`for slot in ctx_arr:`遍历、`len(ctx_arr)`<br>越界访问编译期检查，越界报`GetItemExpr index N out of bounds for tuple with M elements` |

## 调用示例
### 示例一

下面是一个完整Kernel：创建2槽结构体数组，按索引读取/修改标量字段和数组字段元素，并写回输出Tensor。

```python
import pypto_pro.language as pl


@pl.jit()
def struct_array_kernel(out: pl.Tensor[[4], pl.DT_INT32]):
    # 创建 2 槽结构体数组：标量字段 + 数组字段
    run_infos = pl.struct_array(2, "run_info", batch_id=0, innerS1Realsize=[0, 0, 0, 0])

    # 按索引修改标量字段
    run_infos[0].batch_id = 7
    run_infos[1].batch_id = 9

    # 按索引修改数组字段元素
    run_infos[0].innerS1Realsize[3] = 128
    run_infos[1].innerS1Realsize[1] = 64

    # 读回数组字段元素并输出
    with pl.section_vector():
        pl.setval(out, 0, run_infos[0].batch_id)
        pl.setval(out, 1, run_infos[1].batch_id)
        pl.setval(out, 2, run_infos[0].innerS1Realsize[3])
        pl.setval(out, 3, run_infos[1].innerS1Realsize[1])
```

### 示例二

下面是一个完整Kernel：for循环中按循环变量索引读写结构体数组的数组字段元素。

```python
import pypto_pro.language as pl


@pl.jit()
def struct_array_field_kernel(out: pl.Tensor[[4], pl.DT_INT32]):
    # 创建 4 槽结构体数组，含数组字段
    run_infos = pl.struct_array(4, "run_info", batch_id=0, innerS1Realsize=[0, 0, 0, 0])

    with pl.section_vector():
        # 数组字段元素赋值（arr[i].field[j] = val）
        for i in pl.range(0, 4):
            run_infos[i].innerS1Realsize[0] = i * 100

        # 数组字段元素读取（arr[i].field[j]）
        for i in pl.range(0, 4):
            pl.setval(out, i, run_infos[i].innerS1Realsize[0])
```
