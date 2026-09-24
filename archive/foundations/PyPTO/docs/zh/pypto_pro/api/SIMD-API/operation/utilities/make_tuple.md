# pypto_pro.language.make_tuple

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

编译期元组，纯IR不生成C++代码。将多个IR变量按字段名打包，字段访问被常量折叠回原值，零运行时开销。

与`pypto_pro.language.struct`的区别：`pypto_pro.language.struct`生成真实C++ struct（用于跨pipe传递），`pypto_pro.language.make_tuple`不生成C++ 代码（仅用于IR层面聚合变量）。

## 函数原型

```python
pypto_pro.language.make_tuple(field1=val1, field2=val2, ...)
```

## 参数类型

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `field=value` | 输入 | 字段名和对应的IR表达式（关键字参数） |

## 参数范围

| 参数 | 输入/输出 | 说明 |
|---|---|---|
| `field=value` | 输入 | 至少一个关键字参数<br>字段名须为合法标识符<br>字段值支持Kernel内可解析的值表达式，包括Python标量值、Kernel标量表达式、Tensor、Tile、Ptr以及由这些值组成的Python元组（如`(tile0, tile1)`）<br>不支持位置参数<br>不支持`**kwargs`展开 |

## 补充说明

返回一个命名元组对象，支持通过字段名访问打包的变量。字段访问在编译期被常量折叠回原值，不产生运行时开销。

典型使用场景：

1. **函数返回多个值**：将多个Tile或变量打包返回，避免使用全局变量
2. **组织相关变量**：将逻辑上相关的变量分组，提高代码可读性
3. **双缓冲管理**：将ping/pong缓冲区的多个Tile打包，通过字段名访问

与`struct`的选择：

- 需要跨pipe传递（如SSBUF通信）→ 使用`struct`
- 仅在同一pipe内组织变量 → 使用`make_tuple`

## 调用示例

```python
import pypto_pro.language as pl

# 场景 1：标量字段打包 + 字段访问
@pl.jit()
def tuple_scalar_kernel(
    out: pl.Tensor[[2], pl.DT_INT32],
):
    s = pl.struct("TScalar", a=11, b=22)
    with pl.section_vector():
        t = pl.make_tuple(first=s.a, second=s.b)
        pl.setval(out, 0, t.first + t.second)
        pl.setval(out, 1, t.second - t.first)
# 场景 2：for 循环内打包 + struct 字段中转
@pl.jit()
def tuple_in_loop_kernel(
    out: pl.Tensor[[1], pl.DT_INT32],
):
    acc = pl.struct("LoopT", v=0, cur=0)
    with pl.section_vector():
        for i in pl.range(0, 4):
            acc.cur = i
            t = pl.make_tuple(x=acc.cur, y=acc.cur * 10)
            acc.v = acc.v + t.x + t.y
        pl.setval(out, 0, acc.v)
```
