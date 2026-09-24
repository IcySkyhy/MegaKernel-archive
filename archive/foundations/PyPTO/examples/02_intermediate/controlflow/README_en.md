# Runtime Features

This directory demonstrates various advanced features of the PyPTO Runtime, helping developers handle complex execution logic in real-world scenarios.

## Overview

Beyond basic operator computation, real-world applications often need to handle dynamic shapes, conditional statements, and loop control. This directory covers the following content:
- **Dynamic Shapes**: Demonstrates how to handle fluctuating Batch Size or sequence length during inference.
- **Control Flow**: Includes conditional branches inside operators and complex loop structures (`pypto.loop`).
- **Input Order Flexibility**: Demonstrates the JIT kernel's tolerance for input parameter order.

## Sample Code Features

- **`others/dynamic.py`**: Demonstrates the `dynamic_axis` annotation method and enabling support for unaligned dynamic shapes in JIT configuration.
- **`condition/condition.py`**: Demonstrates how to use `if_else` inside kernel functions to build complex logical branches.
- **`loop/`**: Introduces advanced usage of `pypto.loop`, such as loop unrolling and compile-time print debugging.
- **`others/kernel_input.py`**: Demonstrates the JIT kernel's tolerance for input parameter order.

## Code Structure

- **`condition/`**: Conditional branch samples directory. For details, refer to [condition/README_en.md](condition/README_en.md).
- **`loop/`**: Loop control topic directory. For details, refer to [loop/README_en.md](loop/README_en.md).
- **`others/`**: Other Runtime feature samples directory. For details, refer to [others/README_en.md](others/README_en.md).
  - `dynamic.py`: Dynamic shape handling sample.
  - `kernel_input.py`: Input order flexibility sample.

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

### Execute the Script

Run the corresponding Python script directly:

```bash
python3 others/dynamic.py
python3 condition/condition.py
python3 others/kernel_input.py
```

## Precautions

- Loops and conditional statements inside kernels directly affect code generation. For more underlying details, refer to `loop/README.md`.
