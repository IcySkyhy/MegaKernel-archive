# Custom Operator Operations Samples

This directory contains intermediate-level samples for developing custom operators using PyPTO, covering operator composition and best practices for high-performance operator implementation.

## Overview

Custom operator development is one of the core capabilities of PyPTO. Through the samples in this directory, you will learn about:
- **Operator Composition**: How to build complex non-standard activation functions using basic mathematical operators.
- **High-Performance Implementation**: Using Softmax as a sample to demonstrate how to achieve industrial-grade performance through numerically stable algorithms and explicit tiling.

## Sample Features

- **Numerical Stability**: Demonstrates the classic strategy of subtracting the maximum value when handling exponential operations on the NPU.
- **Highly Customizable**: Shows how to compose basic operators such as `exp`, `sum`, and `amax` to implement mainstream model operators like SiLU, GELU, and SwiGLU.
- **PyTorch Integration**: All custom operators include precision comparison logic to ensure consistent behavior with the standard framework.

## Code Structure

- **`activation/`**:
  - `activation.py`: Implements multiple complex activation functions including SiLU, GELU, SwiGLU, and GeGLU.
- **`softmax/`**:
  - `softmax.py`: Provides a detailed step-by-step implementation and optimization of the Softmax operator.

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

Navigate to the corresponding subdirectory and run:

```bash
# Run activation function samples
cd activation
python3 activation.py

# Run Softmax samples
cd ../softmax
python3 softmax.py
```

## Precautions

- When developing operators involving nonlinear transformations (such as Exp, Log), pay attention to data overflow issues.
- The performance of custom operators is significantly affected by the tiling strategy. Tune the tiling according to the shapes in your actual business scenario.
