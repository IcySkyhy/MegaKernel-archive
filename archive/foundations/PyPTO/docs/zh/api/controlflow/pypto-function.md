# pypto.function

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持

## 功能说明

定义一个PyPTO计算函数。在该函数下可添加构建计算图所需要的操作。

## 函数原型

```python
function(name: str, *args, **kwargs) -> Iterator
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                                 |
|--------|-----------|----------------------------------------------------------------------|
| name   | 输入      | 函数的名称，用于标识该计算图。 |
| *args  | 输入      | 从中获取传入的Tensor。 |

## 返回值说明

返回一个上下文管理器，在with语句中使用

## 约束说明

无。

## 调用示例

```python
with pypto.function("main", a, b, c):
    pypto.set_vec_tile_shapes(16, 16)
    for _ in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx_mla_prolog", idx_name="b_idx"):
        c[:] = a + b
```
