# Multi-Function Module Composition Sample

This sample demonstrates how to compose multiple `@pypto.frontend.jit` functions together to build complex computation pipelines and modular neural network architectures.

## Overview

When developing large operators (such as a complete Transformer layer), writing all logic in a single large function hinders maintainability and optimization. This sample demonstrates the following core patterns:
- **Sequential Composition**: Executing multiple JIT functions in sequence.
- **Residual Connection**: Establishing skip connections across different function execution paths.
- **Function Reuse**: Calling the same JIT function multiple times with different inputs.
- **Building Complex Modules**: Progressively building a complete Transformer block from small functional functions (LayerNorm, Linear, Activation).

## Code Structure

- **`function.py`**: Core sample code demonstrating various composition patterns.
  - `test_sequential_functions`: Sequential execution sample.
  - `test_residual_connection`: Residual connection sample.
  - `test_transformer_block`: Comprehensive sample building a complete Transformer block.
  - `test_function_reuse`: Function reuse sample.
- **`multi_jit.py`**: Multi-JIT function composition creation sample, showing how to chain multiple independently compiled JIT functions together.

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
# Run all composition pattern samples
python3 function.py

# List all available samples
python3 function.py --list
```

## Core Pattern Analysis

### 1. Sequential Composition
```python
# Step 1: Normalization
layer_norm([x, gamma, beta], [normed])
# Step 2: Activation
gelu_activation([normed], [activated])
```

### 2. Residual Connection
```python
# 1. Main branch computation
processed = process(x)
# 2. Residual addition
residual_add([x, processed], [output])
```

### 3. Function Reuse
A JIT function compiles on its first call. Subsequent calls directly execute the compiled binary code, even with different tensor values.

## Best Practices
- **Single Responsibility**: Each JIT function should focus on completing a single, independent task.
- **Pre-allocate Memory**: Prepare output tensors before calling functions to avoid repeated memory allocation in loops.
- **Modular Design**: Follow the PyTorch `nn.Module` style, encapsulating common computation logic into reusable functional functions.

## Precautions
- When passing data across functions, ensure consistent data type (Dtype) and device.
- The performance of function composition is typically related to the operator fusion strategy. The PyPTO backend attempts to optimize these compositions.
