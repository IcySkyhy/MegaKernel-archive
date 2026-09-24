# Softmax Operator Development Sample

This sample demonstrates how to implement an efficient Softmax operator from the ground up using the PyPTO framework.

## Overview

Softmax is an extremely commonly used operator in deep learning, especially in the Attention mechanism. This sample details the following implementation steps:
- **Numerically Stable Computation Strategy**: Preventing exponential overflow by subtracting the maximum value (Max Subtraction).
- **Dynamic Shape Support**: Handling variable batch sizes.
- **Explicit Tiling Configuration**: Optimizing NPU hardware execution efficiency.
- **Kernel Loop Processing**: Iterating over different data blocks through loops inside the kernel.

## Core Algorithm Implementation

### 1. Core Computation Logic (`softmax_core`)

```python
def softmax_core(x: pypto.Tensor) -> pypto.Tensor:
    # Find the maximum value along the last dimension
    row_max = pypto.amax(x, dim=-1, keepdim=True)
    # Subtract the maximum value for numerical stability
    sub = x - row_max
    # Compute the exponential
    exp = pypto.exp(sub)
    # Compute the sum of exponentials
    esum = pypto.sum(exp, dim=-1, keepdim=True)
    # Normalize
    return exp / esum
```

### 2. JIT Kernel Wrapper (`softmax_kernel`)

The kernel function manages tiling and looping:

```python
@pypto.frontend.jit
def softmax_kernel(
    x: pypto.Tensor(x_shape, pypto.DT_FP32),
):
    # Set the tiling shape
    pypto.set_vec_tile_shapes(1, 4, 1, 64)
    # Use pypto.loop to process data blocks
    for idx in pypto.loop(b_loop):
        # ... view partitioning and computation ...
```

## Code File Description

- **`softmax.py`**: Contains the complete Softmax implementation code and test verification logic.

## Running the Sample

### Environment Setup

```bash
# Configure the CANN environment variables
# After installation, configure the environment variables. For the actual path of set_env.sh, refer to the following command.
# The environment variable configuration above takes effect only in the current window. You can write the commands into an environment variable configuration file (for example, .bashrc) as needed.

# Default path installation, using the root user as an example (for non-root users, replace /usr/local with ${HOME})
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Set the device ID
export TILE_FWK_DEVICE_ID=0
```

### Execute the Script

```bash
python3 softmax.py
```

## Key Features

- **Numerical Stability**: Achieved through the combination of `amax` and `sub`, which is standard practice for developing high-performance activation operators on the NPU.
- **PyTorch Verification**: The implementation result is compared against `torch.softmax` using `assert_allclose`.
- **High-Performance Tiling**: Demonstrates how to configure the optimal computation tile for the Vector Core.

## Precautions

- Operator performance heavily depends on the `set_vec_tile_shapes` setting. Tune it according to the actual hidden size.
- This sample demonstrates Softmax along `dim=-1`. To compute along other dimensions, adjust the dimension parameters of `amax` and `sum` accordingly.
