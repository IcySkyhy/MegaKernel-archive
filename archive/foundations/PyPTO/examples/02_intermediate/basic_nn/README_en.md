# Neural Network Components

This directory contains intermediate development samples for building common neural network components using PyPTO. These components form the foundation for building large Transformer models.

## Overview

At this stage, you learn how to combine basic operators into neural network layers with specific functionality. This directory covers the following core components:
- **Layer Normalization**: Demonstrates the implementation of standard LayerNorm and RMSNorm, involving mean and variance reduction computation.
- **FFN Module (Feed-Forward Network)**: Implements a complete feed-forward network supporting multiple activation functions (ReLU, GELU, SwiGLU) and dynamic Batch Size handling.

## Sample Code Features

The code in this directory demonstrates the following advanced application features of PyPTO:
- **Modular Design**: Demonstrates how to build reusable network modules by encapsulating `pypto.frontend.jit` functions.
- **Dynamic Shape Support**: Shows how to use `dynamic_axis` to handle varying Batch dimensions in the FFN module.
- **High-Performance Operator Combination**: Demonstrates deep integration of matrix multiplication, element-wise operations, and reduction operations (such as Sum, Max).

## Code Structure

- **`layer_normalization/`**:
  - `layer_norm.py`: Contains the core implementation of LayerNorm and RMSNorm along with precision verification.
- **`ffn/`**:
  - `ffn_module.py`: Core implementation of the FFN module and various scenario tests.

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

Enter the corresponding subdirectory and run the script:

```bash
# Run the LayerNorm sample
cd layer_normalization
python3 layer_norm.py

# Run the FFN sample
cd ../ffn
python3 ffn_module.py
```

## Learning Suggestions

1. Start with `layer_normalization` to master the basic reduction computation pattern.
2. Then learn `ffn` to understand how to integrate matrix multiplication with complex activation function logic.
3. Refer to the sub-README in each directory for more detailed algorithm descriptions.
