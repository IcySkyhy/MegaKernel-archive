# Tiling配置

合理设置TileShape对于优化算子性能至关重要。TileShape定义了数据在硬件的不同计算单元中的切分方式，影响数据搬运和计算效率。通过合理设置TileShape，可以显著提升计算性能，减少数据搬运开销，实现高效计算。

## 原理介绍

TileShape设置的核心在于根据硬件资源和计算需求，合理划分数据块，以最大化利用硬件资源，减少数据搬运开销，从而提升计算性能。

- 向量计算：在向量计算中，set\_vec\_tile\_shapes用于设置向量数据在各维度上的切分大小。合理的切分可以使数据充分利用统一缓冲区（Unified Buffer，UB），在向量计算单元上高效处理。
- 矩阵计算：在矩阵计算中，将矩阵相乘形状变化记为\(m, k\) x \(k, n\) = \(m, n\)，set\_cube\_tile\_shapes用于依次设置矩阵在m、k、n维度上的切分大小。合理的切分可以充分利用L0和L1缓冲区，减少数据搬运开销。

## Vector计算的Tiling配置

set\_vec\_tile\_shapes用于设置向量计算中各维度的TileShape。

```python
# 设置向量计算的TileShape
pypto.set_vec_tile_shapes(1, 1, 8, 8)
# 获取并打印设置的TileShape
print(pypto.get_vec_tile_shapes())  # 输出: [1, 1, 8, 8]
```

pypto.set\_vec\_tile\_shapes\(1, 1, 8, 8\)表示该向量有四个维度，每个维度分别按照1, 1, 8, 8的大小进行切分，并按 \(1, 1, 8, 8\)的切分大小将原向量搬移到UB上进行运算。

实际用例如下：

```python
@pypto.frontend.jit
def compute_with_vec_tile_shapes_kernel(
    a: pypto.Tensor((32, 32), pypto.DT_FP32),
    b: pypto.Tensor((32, 32), pypto.DT_FP32),
    out: pypto.Tensor((32, 32), pypto.DT_FP32),
    set_shapes: tuple
):
    pypto.set_vec_tile_shapes(*set_shapes)
    out[:] = pypto.add(a, b)

def compute_with_vec_tile_shapes_op(a: torch.Tensor, b: torch.Tensor, set_shapes: tuple, dynamic: bool = False) -> torch.Tensor:
    # 直接传入torch tensor调用
    out = torch.empty_like(a)
    compute_with_vec_tile_shapes_kernel(a, b, out, set_shapes)
    return out

def test_set_vec_tile_shapes_basic():
    ...
    a = torch.tensor([[[1, 2, 3],
                       [1, 2, 3]]], dtype=dtype, device=f'npu:{device_id}')
    b = torch.tensor([[[4, 5, 6],
                       [4, 5, 6]]], dtype=dtype, device=f'npu:{device_id}')
    expected = torch.tensor([[[5, 7, 9],
                            [5, 7, 9]]], dtype=dtype, device=f'npu:{device_id}')
    set_shapes = (1, 2, 8)
    out = compute_with_vec_tile_shapes_op(a, b, set_shapes)
    assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
```

上述用例展示了一个在简单的向量加法场景下set\_vec\_tile\_shapes的应用。

需要说明的是，通常设置不同的TileShape不影响向量的计算结果，但是会影响向量计算的运行时间，如下述用例所示：

```python
@pypto.frontend.jit
def compute_with_vec_specific_tile_shapes_kernel(
    a: pypto.Tensor((4, 32, 64, 256), pypto.DT_FP32),
    b: pypto.Tensor((4, 32, 64, 256), pypto.DT_FP32),
    out: pypto.Tensor((4, 32, 64, 256), pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(1, 2, 4, 128)
    out[:] = pypto.add(a, b)

@pypto.frontend.jit
def compute_with_vec_another_tile_shapes_kernel(
    a: pypto.Tensor((4, 32, 64, 256), pypto.DT_FP32),
    b: pypto.Tensor((4, 32, 64, 256), pypto.DT_FP32),
    out: pypto.Tensor((4, 32, 64, 256), pypto.DT_FP32),
):
    pypto.set_vec_tile_shapes(2, 4, 8, 256)
    out[:] = pypto.add(a, b)

def compute_with_vec_specific_tile_shapes_op(a: torch.Tensor, b: torch.Tensor, dynamic: bool = False) -> torch.Tensor:
    out = torch.empty_like(a)
    compute_with_vec_specific_tile_shapes_kernel(a, b, out)
    return out

def compute_with_vec_another_tile_shapes_op(a: torch.Tensor, b: torch.Tensor, dynamic: bool = False) -> torch.Tensor:
    out = torch.empty_like(a)
    compute_with_vec_another_tile_shapes_kernel(a, b, out)
    return out

def test_set_vec_different_tile_shapes_runtime():
    ...
    a = torch.randn((4, 32, 64, 256), dtype=dtype, device=f'npu:{device_id}')
    b = torch.randn((4, 32, 64, 256), dtype=dtype, device=f'npu:{device_id}')
    TEST_TIME = 1
    start = time.perf_counter()
    for _ in range(TEST_TIME):
        out1 = compute_with_vec_specific_tile_shapes_op(a, b)
    runtime_1 = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(TEST_TIME):
        out2 = compute_with_vec_another_tile_shapes_op(a, b)
    runtime_2 = time.perf_counter() - start
    print(f"runtime_1(pypto.set_vec_tile_shapes(1, 2, 4, 128)): {runtime_1}")
    print(f"runtime_2(pypto.set_vec_tile_shapes(2, 4, 8, 256)): {runtime_2}")
```

在该示例中，对于两个shape为\(4, 32, 64, 256\)的向量相加，set\_vec\_tile\_shapes\(1, 2, 4, 128\)的运行时间明显比set\_vec\_tile\_shapes\(2, 4, 8, 256\)要长。

完整样例请参考：[tiling_config.py](../../../../examples/01_beginner/tiling/tiling_config.py)。

## Cube计算的Tiling配置

set\_cube\_tile\_shapes用于设置矩阵计算中各矩阵在m、k、n维度上的TileShape。
具体优化切分配置可参考[Matmul高性能编程](../debug/matmul_performance_guide.md)。

```python
# 设置Cube计算的TileShape
pypto.set_cube_tile_shapes([16, 16], [256, 512], [128, 128], enable_split_k=False)
# 获取并打印设置的TileShape
print(pypto.get_cube_tile_shapes())  # 输出: [[16, 16], [256, 512, 512], [128, 128], False]
```

pypto.set\_cube\_tile\_shapes\(\[16, 16\], \[256, 512\], \[128, 128\], enable\_split\_k=False\)：将矩阵相乘形状变化记为\(m, k\) x \(k, n\) = \(m, n\)，这里三个列表分别设置的是矩阵m、k、n维度的切分大小，每个列表的两个元素，第一个元素是设置L0的切分大小，第二个元素设置的是L1的切分大小。其中enable\_split\_k参数为是否开启多核切K功能，默认为False。对于M、N较小而K轴较大的场景，仅在M、N轴做分核可能无法用满核导致整体性能较差，此时可以设置`enable_split_k`为True以使能K轴分核。

实际用例如下：

```python
@pypto.frontend.jit
def compute_with_cube_tile_shapes_kernel(
    a: pypto.Tensor((64, 64), pypto.DT_FP32),
    b: pypto.Tensor((64, 64), pypto.DT_FP32),
    out: pypto.Tensor((64, 64), pypto.DT_FP32),
    set_shapes: list
):
    pypto.set_cube_tile_shapes(*set_shapes)
    out[:] = pypto.matmul(a, b, a.dtype)

def compute_with_cube_tile_shapes_op(a: torch.Tensor, b: torch.Tensor, set_shapes: list, dynamic: bool = False) -> torch.Tensor:
    # 直接传入torch tensor调用
    out = torch.empty((64, 64), dtype=a.dtype, device=a.device)
    compute_with_cube_tile_shapes_kernel(a, b, out, set_shapes)
    return out

def test_set_cube_tile_shapes_basic():
    ...
    a = torch.tensor([[1, 2], [3, 4]], dtype=dtype, device=f'npu:{device_id}')
    b = torch.tensor([[5, 6], [7, 8]], dtype=dtype, device=f'npu:{device_id}')
    expected = torch.tensor([[19, 22], [43, 50]], dtype=dtype, device=f'npu:{device_id}')
    set_shapes = [[32, 32], [64, 64], [64, 64]]
    out = compute_with_cube_tile_shapes_op(a, b, set_shapes)
    assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=1e-3, atol=1e-3)
```

上述用例展示了一个在简单的矩阵乘法场景下set\_cube\_tile\_shapes的应用。

需要说明的是，通常设置不同的TileShape不影响矩阵的计算结果，但是会影响矩阵计算的运行时间，如下述用例所示：

```python
import pypto
import torch
import time

@pypto.frontend.jit
def compute_with_cube_specific_tile_shapes_kernel(
    a: pypto.Tensor((4, 64, 512), pypto.DT_FP32),
    b: pypto.Tensor((4, 128, 512), pypto.DT_FP32),
    out: pypto.Tensor((4, 64, 128), pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([32, 32], [32, 32], [32, 32])
    out[:] = pypto.matmul(a, b, a.dtype, b_trans=True)

@pypto.frontend.jit
def compute_with_cube_another_tile_shapes_kernel(
    a: pypto.Tensor((4, 64, 512), pypto.DT_FP32),
    b: pypto.Tensor((4, 128, 512), pypto.DT_FP32),
    out: pypto.Tensor((4, 64, 128), pypto.DT_FP32),
):
    pypto.set_cube_tile_shapes([64, 64], [128, 128], [128, 128])
    out[:] = pypto.matmul(a, b, a.dtype, b_trans=True)

def compute_with_cube_specific_tile_shapes_op(a: torch.Tensor, b: torch.Tensor, dynamic: bool = False) -> torch.Tensor:
    out = torch.empty((4, 64, 128), dtype=a.dtype, device=a.device)
    compute_with_cube_specific_tile_shapes_kernel(a, b, out)
    return out

def compute_with_cube_another_tile_shapes_op(a: torch.Tensor, b: torch.Tensor, dynamic: bool = False) -> torch.Tensor:
    out = torch.empty((4, 64, 128), dtype=a.dtype, device=a.device)
    compute_with_cube_another_tile_shapes_kernel(a, b, out)
    return out

def test_set_cube_different_tile_shapes_runtime():
    ...
    a = torch.randn((4, 64, 512), dtype=dtype, device=f'npu:{device_id}')
    b = torch.randn((4, 128, 512), dtype=dtype, device=f'npu:{device_id}')
    TEST_TIME = 1
    start = time.perf_counter()
    for _ in range(TEST_TIME):
        out1 = compute_with_cube_specific_tile_shapes_op(a, b)
    runtime_1 = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(TEST_TIME):
        out2 = compute_with_cube_another_tile_shapes_op(a, b)
    runtime_2 = time.perf_counter() - start
    print(f"runtime_1(pypto.set_cube_tile_shapes([32, 32], [32, 32], [32, 32])): {runtime_1}")
    print(f"runtime_2(pypto.set_cube_tile_shapes([64, 64], [128, 128], [128, 128])): {runtime_2}")
```

在该示例中，对于两个shape分别为\(4, 64, 512\)和的\(4, 512, 128\)的矩阵相乘，set\_cube\_tile\_shapes\(\[32, 32\], \[32, 32\], \[32, 32\]\)的运行时间明显比set\_cube\_tile\_shapes\(\[64, 64\], \[128, 128\], \[128, 128\]\)要长。

完整样例请参考：[tiling_config.py](../../../../examples/01_beginner/tiling/tiling_config.py)。

## 使用约束

- 设置TileShape参数时，需要满足约束条件，应与需要处理的Tensor的Shape维度数量和大小相匹配，且数值不能过小或过大。

    TileShape数值设置不能过小，过小的TileShape会导致切分次数TensorShape/TileShape（即TensorShape与TileShape每个维度比值的乘积）过大，从而使得表达式在线循环展开的次数过大，这可能导致表达式表编译失败，并会增加运行时的头开销。表达式表的大小与在线循环展开次数以及算子输入个数有关，建议控制 \(TensorShape/TileShape\)\*\(1+算子输入个数\)的值小于18000。

    TileShape数值设置不能过大，过大的TileShape会超出相应硬件（缓冲区）存储的大小，应控制切分后的数据大小（数据类型大小与切分后数据各维度大小乘积）不大于对应硬件单元存储容量。

    此外，set\_vec\_tile\_shapes的维度数量不大于5，尾轴切分大小满足32B对齐。set\_cube\_tile\_shapes要求kL0、kL1、nL0、nL1均满足32字节对齐。详细的配置要求请参见相关接口文档。

- 设置TileShape会影响上板时间。一般来说，越能充分利用硬件单元的容量，即一次计算的数据量越大，运行时间就越短。然而TileShape参数设置得越大并不一定意味着上板运行会更快，还需要考虑数据搬运等环节的开销。

## 其他操作

- 性能观察：可以通过性能分析工具（如泳道图）观察不同TileShape下的性能，从而评估TileShape设置的合理性，获取当前场景下最优TileShape。
- 精度影响：除非设置极端值，一般不影响精度（框架不希望出现的行为）。
