# Advanced NN Architectures Samples

This directory contains advanced neural network architecture samples implemented using PyPTO, focusing on the core computational components of large language models (LLMs).

## Overview

At this stage, developers face more complex computation graphs and stricter performance requirements. This directory currently focuses on:
- **Attention Mechanism**: Implements scaled dot-product attention and a complete multi-head attention module.

## Sample Features

- **Complex Tensor Transformations**: Demonstrates how to split and merge multi-head attention through frequent `transpose` and `reshape` operations.
- **Extreme Performance Optimization**: Shows how to configure optimal `cube_tile_shapes` for large-scale matrix multiplications in the attention mechanism.
- **Dynamic Batch and Sequence Length**: Demonstrates how to handle varying input scales during real-time inference requests through dynamic axis marking.

## Code Structure

- **`attention/`**:
  - `attention.py`: Contains the complete attention mechanism implementation, configuration class, and precision comparison against native PyTorch operators.

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
cd attention
python3 attention.py
```

## Learning Suggestions

1. The attention mechanism is the cornerstone of modern deep learning. Read `attention/README.md` in depth to understand the algorithm details.
2. Try modifying the tiling configuration in `attention.py` and observe its impact on runtime performance.
3. Refer to the real model code in the `models` directory to understand how to apply the components in this directory to industrial-grade projects.
