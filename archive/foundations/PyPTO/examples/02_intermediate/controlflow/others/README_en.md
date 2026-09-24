# Other Runtime Features Sample

This directory contains samples of other advanced runtime features in PyPTO.

## Overview

In addition to loops and conditional branches, PyPTO provides the following runtime features:
- **Dynamic Shapes**: Handling fluctuating batch sizes or sequence lengths during inference.
- **Kernel Input Order**: The JIT kernel tolerance for the order of input parameters.

## Code Structure

- **`dynamic.py`**: Dynamic shape handling samples, demonstrating the `dynamic_axis` marking method and the View/Assemble tiling strategy under dynamic dimensions.
  - `test_dynamic_mul`: Basic dynamic batch dimension.
  - `test_dynamic_partial`: Partial dynamic dimensions.
  - `test_dynamic_attention`: Multi-head attention with dynamic batch.
  - `test_dynamic_multi_dim`: Multiple dynamic dimensions.
- **`kernel_input.py`**: Input order flexibility samples, demonstrating the JIT kernel tolerance for parameter order.

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
# Run dynamic shape samples
python3 dynamic.py

# Run kernel input order samples
python3 kernel_input.py
```

## Precautions

- Mark only the necessary axes as dynamic dimensions and keep other dimensions as concrete values for better compile-time optimization.
- When using dynamic shapes, use `pypto.view` / `pypto.assemble` for explicit tiling management.
