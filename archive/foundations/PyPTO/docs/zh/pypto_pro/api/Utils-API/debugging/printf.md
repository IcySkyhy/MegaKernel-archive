# pypto_pro.language.printf

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

在Kernel运行时按格式串打印标量值或指针，用于调试。格式串遵循C `printf`语法的受限子集。

该接口通过设备侧打印机制输出，具体查看位置由运行环境的CANN日志配置决定。`printf`运行在S（标量）流水上，有显著运行时开销，仅用于调试，生产环境应移除。

## 函数原型

```python
pypto_pro.language.printf(format_str, *args, *, loc=False)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `format_str` | 输入 | 格式串，必须是编译时常量字符串 |
| `*args` | 输入 | 要打印的标量值或Ptr指针，数量和类型须与格式串中的格式说明符匹配 |
| `loc` | 输入 | 可选，是否在输出前打印源文件/行号 |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `format_str` | 输入 | 编译时常量字符串，不能是运行时变量<br>支持的格式说明符：`%[flags][width][.precision]conversion`<br>flags：`-`、`+`、` `（空格）、`#`、`0`<br>conversion：`d`/`i`（有符号整数）、`u`（无符号整数）、`x`（十六进制）、`f`（浮点）、`p`（指针，接受Ptr）<br>不支持的conversion：`%%`（百分号本身）、`%s`（字符串）、`%c`（字符）<br>不支持长度修饰符：如`%lld`、`%ld`、`%hd`等 |
| `*args` | 输入 | 支持Python `int`、Python `float`、Python `bool`以及Kernel标量表达式；使用`%p`时还支持Ptr指针<br>`%d`/`%i`：接受有符号整型、BOOL、INDEX，拒绝无符号整型<br>`%u`：接受无符号整型、BOOL、INDEX，拒绝有符号整型<br>`%x`：接受无符号整型、INDEX，拒绝有符号整型和BOOL<br>`%f`：仅接受FP32，拒绝FP16/BF16和整型<br>`%p`：仅接受Ptr指针 |
| `loc` | 输入 | `True`或`False`（默认）<br>为`True`时在输出前打印调用位置的源文件和行号 |

## 流水类型

S（标量流水）。

## 调用示例

```python
import pypto_pro.language as pl
# 打印整数
pl.printf("flag=%d, offset=%d\n", flag, offset)

# 打印浮点（仅 FP32）
pl.printf("value=%+08.3f\n", value_f32)

# 打印十六进制
pl.printf("addr=%#08x\n", addr_u32)

# 纯文本（无参数）
pl.printf("reached checkpoint A\n")

# 带源码位置
pl.printf("debug: i=%d\n", i, loc=True)
```

示例输出：

| 调用 | 输出（假设flag = 1, offset = 32, value_f32 = 3.14, addr_u32 = 0x1234, i = 5） |
|---|---|
| `pl.printf("flag=%d, offset=%d\n", flag, offset)` | `flag=1, offset=32` |
| `pl.printf("value=%+08.3f\n", value_f32)` | `value=+003.140` |
| `pl.printf("addr=%#08x\n", addr_u32)` | `addr=0x001234` |
| `pl.printf("reached checkpoint A\n")` | `reached checkpoint A` |
| `pl.printf("debug: i=%d\n", i, loc=True)` | `debug: i=5`（loc = True时额外附带源码位置） |
