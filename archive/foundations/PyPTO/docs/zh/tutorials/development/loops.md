# 循环和数据切分

在前面的介绍中，经过Tiling配置后，Tensor的输入和输出会在核内按照Tiling配置方案进行切分处理。例如，如果原始Tensor为\(1, 32, 1, 256\)，Tiling方案配置为\(1, 4, 1, 64\)后，核内将按照\(1, 4, 1, 64\)进行切分和循环处理。

如果数据量增大，Tensor配置变为\(32, 32, 1, 256\)，第一个轴（shape\[0\]或Batch轴）可以通过pypto.loop增加循环逻辑，使框架能够将多个batch展开并行处理。

## 基础循环结构

```python
SHAPE = (32, 32, 1, 256)

@pypto.frontend.jit
def add_kernel(
    input0: pypto.Tensor(SHAPE, pypto.DT_FP32),
    input1: pypto.Tensor(SHAPE, pypto.DT_FP32),
    out: pypto.Tensor(SHAPE, pypto.DT_FP32),
    val: int
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    #calculate the loop parameters
    b, n, s, d = SHAPE
    tile_b = 1
    b_loop = b // tile_b

    for idx in pypto.loop(b_loop):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b
        t0_sub = input0[b_offset:b_offset_end, ...]
        t1_sub = input1[b_offset:b_offset_end, ...]
        t3_sub = t0_sub + t1_sub
        out[b_offset:b_offset_end, ...] = t3_sub + val
```

循环使用的pypto.loop的全接口入参如下：

```python
for idx in pypto.loop(start, end, step, name="label", idx_name="idx_label", submit_before_loop=False)
```

- 参数说明：
    - start, end, step：可选参数，支持灵活配置循环范围。
    - name, idx\_name：循环标识和索引变量名称，用于调试。
    - submit\_before\_loop：控制循环执行顺序。

也可以按需简化成：

```python
for idx in pypto.loop(start, end, step)   #不带loop的名字标签
for idx in pypto.loop(start, end)         # 默认step=1
for idx in pypto.loop(end)                # 默认start=0， step=1
```

slicing语法糖的完整样例请参考：[loop.py](../../../../examples/02_intermediate/controlflow/loop/loop.py)。

## 使用view/assemble接口处理循环结构

上述示例展示了如何在循环中使用Python的切片操作 \[:, :, :\] 来提取小块数据进行计算。计算完成后，数据会被输出。此外，也可以利用pypto.view接口提取小块数据进行计算，计算完成后，再通过pypto.assemble将数据组装并输出。

```python
SHAPE = (32, 32, 1, 256)

@pypto.frontend.jit
def add_kernel(
    input0: pypto.Tensor(SHAPE, pypto.DT_FP32),
    input1: pypto.Tensor(SHAPE, pypto.DT_FP32),
    out: pypto.Tensor(SHAPE, pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    #calculate the loop parameters
    b, n, s, d = SHAPE
    tile_b = 1
    b_loop = b // tile_b

    for idx in pypto.loop(b_loop):
        b_offset = idx * tile_b
        t0_sub = pypto.view(input0, [tile_b, n, s, d], [b_offset, 0, 0, 0])
        t1_sub = pypto.view(input1, [tile_b, n, s, d], [b_offset, 0, 0, 0])
        t3_sub = t0_sub + t1_sub
        pypto.assemble(t3_sub, [b_offset, 0, 0, 0], out)
```

view/assemble接口完整样例请参考：[add_scalar_loop_view_assemble.py](../../../../examples/01_beginner/transform/add_scalar_loop_view_assemble.py)。

## 数据依赖与循环顺序

默认情况下，pypto.loop会展开并分发到多核并行处理，适用于无数据依赖的场景。如果循环之间存在数据依赖（如前一个循环的输出是下一个循环的输入），需设置submit\_before\_loop = True，确保每个循环迭代的结果写回Tensor后，再启动下一个循环：

```python
for idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="idx", submit_before_loop=True):
```
