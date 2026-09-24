# 循环中使用Python print函数打印

## 问题现象描述

```python
@pypto.frontend.jit
def add_kernel_0(a, b, c):
    for i in pypto.loop(20):
        print("i = ", i)
        c[:] = a + b
>>>
i = 0

@pypto.frontend.jit
def add_kernel_1(a, b, c):
    for i in pypto.loop(20):
        print("i = ", i)
        if pypto.cond(i == 0):
            c[:] = a + b
        else:
            c[:] = a - b
>>>
i = 0
i = 1
```

## 可能原因

用户算子描述的是构图过程，而非实际的执行逻辑。在构图阶段，Loop执行仅用于遍历所有执行路径。

- 示例代码`add_kernel_0`中仅有一条执行路径，因此Loop仅执行一次。
- `add_kernel_1`中存在if/else两条路径，因此Loop执行两次。

## 处理步骤

N/A
