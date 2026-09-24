# Compute Operations

This directory contains usage samples of various compute operators in PyPTO, including element-wise operations, matrix multiplication, and reduction operations.

## Overview

The compute operator samples cover the following content:
- **Element-wise Operations**: Arithmetic operations applied independently to each element of a tensor, such as addition, subtraction, multiplication, division, absolute value, exponent, logarithm, and so on.
- **Matrix Multiplication**: Matrix multiplication operations under various configurations.
- **Reduction Operations**: Operations that reduce a tensor along specified dimensions, such as sum, maximum, minimum, and so on.

## Code File Description

- **`elementwise_ops.py`**: Element-wise operator samples covering `abs`, `add`, `clip`, `div`, `exp`, `expm1`, `log`, `mul`, `neg`, `pow`, `rsqrt`, `sqrt`, `ceil`, `floor`, `trunc`, `round`, `sub`.
- **`matmul_ops.py`**: Matrix multiplication samples covering basic matrix multiplication, batched matrix multiplication, broadcast matrix multiplication, matrix multiplication with transpose, and matrix multiplication with Bias.
- **`reduce_ops.py`**: Reduction operator samples covering `amax`, `amin`, `maximum`, `minimum`, `sum`.

## How To Run

### Environment Preparation

```bash
# Configure CANN environment variables
# After installation, configure the environment variables. Execute the following command based on the actual path of set_env.sh.
# The above environment variable configuration only takes effect in the current window. You can write the above commands into the environment variable configuration file (such as .bashrc) as needed.

# Default path installation, using root user as a sample (for non-root users, replace /usr/local with ${HOME})
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Set device ID
export TILE_FWK_DEVICE_ID=0
```

### Run the Samples

Each script supports listing all test cases or running a specific test case:

```bash
# Run all element-wise operation samples
python3 elementwise_ops.py

# List all available element-wise operation test cases
python3 elementwise_ops.py --list

# Run a specific test case
python3 elementwise_ops.py abs::test_abs_basic
```

## Operator Feature Description

### Element-wise Operations

Supports broadcasting mechanism and scalar operations.

```python
@pypto.frontend.jit()
def add_example(
    a: pypto.Tensor(shape, dtype),
    b: pypto.Tensor(shape, dtype),
    out: pypto.Tensor(shape, dtype)
):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.add(a, b)

```

### Matrix Multiplication

Uses Cube Tiling for efficient computation and supports specifying output data types.

```python
@pypto.frontend.jit()
def matmul_example(
    a: pypto.Tensor((M, K), pypto.DT_BF16),
    b: pypto.Tensor((K, N), pypto.DT_BF16),
    out: pypto.Tensor((M, N), pypto.DT_BF16)
):
    pypto.set_cube_tile_shapes([32, 32], [64, 64], [64, 64])
    out[:] = pypto.matmul(a, b, out_dtype=pypto.DT_BF16)
```

### Reduction Operations

Supports specifying the dimension (dim) and whether to keep the dimension (keepdim).

```python
@pypto.frontend.jit()
def sum_example(
    x: pypto.Tensor(x_shape, dtype),
    out: pypto.Tensor(out_shape, dtype)
):
    pypto.set_vec_tile_shapes(2, 8)
    out[:] = pypto.sum(x, dim=0, keepdim=True)
```

## Precautions

- When performing matrix multiplication, explicitly set the Cube Tile shapes for optimal performance.
- Reduction operations typically involve data interaction across Tiles. Pay attention to the Tiling partitioning strategy.
- All samples include comparison verification with native PyTorch operators to ensure the accuracy of computation results.
