# Loop Features Sample

This sample demonstrates the advanced features of loop control in PyPTO and their usage rules in operator kernels.

## Overview

In PyPTO JIT kernels, loops not only iterate over data but also involve compile-time optimization and code generation. This sample covers the following key points:
- **Basic Loop Usage**: Using the `start`, `end`, and `step` parameters.
- **Loop Start and End Determination**: How to determine the start and end of a loop.
- **Loop Unroll**: Using the `unroll` interface to optimize performance.
- **Operator Position Rule**: Operators must usually be placed in the innermost loop to ensure correct code generation.
- **Compile-Time Print Feature**: Understanding the `print` behavior inside kernel loops (executed only at compile time).

## Code Structure

- **`loop.py`**: Contains detailed samples and verification logic for loop features.
  - `test_loop_basic`: Basic loop usage (start/stop/step).
  - `test_loop_compile_phase_print`: Compile-time print feature.
  - `test_add_scalar_loop`: Scalar addition loop sample.
  - `test_add_scalar_loop_dyn_axis`: Scalar addition loop sample under dynamic axis.

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
# Run all loop feature samples
python3 loop.py

# List all available test cases
python3 loop.py --list
```

## Core Concepts and Rules

### 1. Basic Usage
```python
for i in pypto.loop(start=0, end=10, step=1):
    # Loop body logic
```

### 2. Operator Position Rule
To ensure that computation instructions are correctly tiled and generate high-performance NPU code, place computation operators (such as `add`, `matmul`, and so on) in the innermost loop of the kernel.

### 3. Compile-Time Print
When you use `print()` directly inside a `pypto.loop`, the print behavior occurs during the compilation phase, not the execution phase. This means it cannot print runtime tensor values and can only assist in debugging the compilation path.

### 4. Loop Unroll
For loops with a small number of iterations, you can use the unroll technique to reduce branch overhead.

## Precautions
- The loop `step` must be a constant.
- The loop structure inside a kernel directly affects the execution of the tiling strategy. Complex loop nesting may require more fine-grained tiling configuration.
