# Custom Activation Function Sample

This sample demonstrates how to compose basic PyPTO operators to implement custom, complex activation functions.

## Overview

In modern Transformer architectures (such as LLaMA, GPT, and PaLM), non-standard activation functions are commonly used. This sample demonstrates the PyPTO implementation of the following activation functions:
- **SiLU (Swish)**: `x * sigmoid(x)`.
- **GELU**: An approximate implementation of the Gaussian Error Linear Unit.
- **SwiGLU**: `Swish(gate) * up`, a type of gated linear unit.
- **GeGLU**: `GELU(gate) * up`.

## Code Structure

- **`activation.py`**: Contains the implementation logic for various custom activation functions and their comparative verification against native PyTorch operators.

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
# Run all activation function samples
python3 activation.py

# List all available samples
python3 activation.py --list
```

## Implementation Pattern: Operator Composition

PyPTO allows you to compose operators like writing mathematical formulas. Using **SiLU** as a sample:

```python
def silu_activation(x: pypto.Tensor) -> pypto.Tensor:
    # 1. Compute sigmoid(x) = 1 / (1 + exp(-x))
    x_neg = pypto.mul(x, -1.0)
    exp_neg = pypto.exp(x_neg)
    sigmoid = pypto.div(1.0, pypto.add(exp_neg, 1.0))

    # 2. Compute x * sigmoid(x)
    return pypto.mul(x, sigmoid)
```

## Key Technical Points

- **Seamless Integration**: Custom activation functions can be used inside `@pypto.frontend.jit` kernels just like built-in operators.
- **Gating Mechanism**: Demonstrates the typical pattern for handling dual-input (Gate and Up) operators, which is very common in modern large models.
- **Precision Verification**: All operators are compared against the standard PyTorch implementation using `assert_allclose`.

## Precautions
- **Operator Fusion**: PyPTO fuses these composed operators as much as possible in the backend to reduce unnecessary memory reads and writes.
- **Data Type**: Develop with BF16 precision to ensure accuracy and benefit from hardware acceleration.
- **Tiling**: Activation functions are typically element-wise operations. The tile shape should fill the vector processing unit as much as possible.
