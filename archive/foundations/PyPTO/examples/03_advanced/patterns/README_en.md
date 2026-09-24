# Advanced Design Patterns Samples

This directory showcases design patterns recommended for developing large-scale, industrial-grade operator libraries.

## Overview

As operator logic complexity increases, maintaining code readability, maintainability, and execution efficiency becomes critical. This directory mainly demonstrates:
- **Multi-Function Modules**: How to build complex computation pipelines (such as a complete Transformer block) by composing multiple independent `pypto.frontend.jit` functions.

## Sample Features

- **Modular Construction**: Shows how to compose LayerNorm, Linear, and Activation functions into a larger module, like building blocks.
- **Residual Connection**: Demonstrates an efficient way to implement skip connections across multi-function call paths.
- **Function Reuse**: Shows how a compiled JIT function can be called multiple times with different data inputs without recompilation.

## Code Structure

- **`function/`**:
  - `function.py`: Demonstrates multiple design patterns including sequential composition, residual connection, and function reuse.
  - `multi_jit.py`: Demonstrates the creation and invocation pattern for multiple JIT functions.

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
cd function
python3 function.py
```

## Precautions

- When composing multiple functions, pre-allocate output tensors. This is critical for reducing memory fragmentation in large model inference.
- The PyPTO backend automatically optimizes the combined execution of these functions to minimize unnecessary memory reads and writes.
