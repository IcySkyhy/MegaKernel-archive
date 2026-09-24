# 条件与分支

条件和分支用于在程序中实现条件判断，从而根据不同的条件执行不同的代码逻辑。编程框架支持两类条件与分支功能：

- 静态条件分支：在编译时配置条件分支，生成执行时固定的指令，可通过多个jit生成不同的kernel。
- 动态条件分支：在运行时判断条件及分支，执行对应的功能。

## 静态条件分支

```python
# 使用入参add1_flag=False生成kernel
@pypto.frontend.jit
def add_kernel_false(
    input0: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    input1: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    val: int
):
    add_core(input0, input1, output, val, False)

#使用入参add1_flag=True生成kernel
@pypto.frontend.jit
def add_kernel_true(
    input0: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    input1: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    val: int
):
    add_core(input0, input1, output, val, True)
```

代码示例：

```python
def add_core(input0: pypto.Tensor, input1: pypto.Tensor, output: pypto.Tensor, val: int, add1_flag: bool = False):
    # Tiling配置与循环逻辑
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    #calculate the loop parameters
    b = input0.shape[0]
    tile_b = 1
    b_loop = b // tile_b

    for idx in pypto.loop(b_loop):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b
        t0_sub = input0[b_offset:b_offset_end, ...]
        t1_sub = input1[b_offset:b_offset_end, ...]
        t3_sub = t0_sub + t1_sub
        if add1_flag:
            output[b_offset:b_offset_end, ...] = t3_sub + val
        else:
            output[b_offset:b_offset_end, ...] = t3_sub
```

该用例在add\_kernel函数中增加了一个可选参数add1\_flag，并使用该参数进行不同的处理。如果add1\_flag为True，则在输出结果上加上参数val；反之，则直接输出前一个处理的结果。

完整样例请参考：[condition.py](../../../../examples/02_intermediate/controlflow/condition/condition.py)。

## 动态条件分支

运行时判断条件及分支，执行对应的功能。核心接口包括：

- `pypto.cond`\(condition\)：运行时判断条件。
- `pypto.is_loop_begin`\(idx\)：判断是否为循环首个迭代。
- `pypto.is_loop_end`\(idx\)：判断是否为循环最后一个迭代。

```python
@pypto.frontend.jit
def add_kernel(
    input0: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    input1: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    output: pypto.Tensor([pypto.DYNAMIC, 4, 1, 64], pypto.DT_FP32),
    val: int
):
    ...
    for idx in pypto.loop(b_loop):
        t3_sub = t0_sub + t1_sub
        if idx < 2:  # 动态条件判断
            output[b_offset:b_offset_end, ...] = t3_sub + val
        else:
            output[b_offset:b_offset_end, ...] = t3_sub

        # 或者基于循环位置的条件
        if pypto.is_loop_begin(idx):
            output[b_offset:b_offset_end, ...] = t3_sub + val
        elif pypto.is_loop_end(idx):
            output[b_offset:b_offset_end, ...] = t3_sub + val + 1
        else:
            output[b_offset:b_offset_end, ...] = t3_sub
```

完整样例请参考：

[condition.py](../../../../examples/02_intermediate/controlflow/condition/condition.py)
