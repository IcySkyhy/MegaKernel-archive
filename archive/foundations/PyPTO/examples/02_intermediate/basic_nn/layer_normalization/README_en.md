# Layer Normalization

This sample demonstrates how to implement commonly used normalization layers in deep learning using PyPTO, including standard LayerNorm and RMSNorm.

## Overview

Normalization is a key component in the Transformer architecture. This sample covers the following content:
- **LayerNorm**: Standard layer normalization with mean and variance computation.
- **RMSNorm**: Root mean square normalization, a simplified variant of LayerNorm commonly used in large models such as LLaMA.
- **Static Shapes**: Efficient implementation with a fixed Batch Size.
- **Dynamic Shapes**: Implementation supporting variable Batch Size.

## Code Structure

- **`layer_norm.py`**: Contains the core implementation of LayerNorm and RMSNorm along with their tests.

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

```bash
# Run all normalization samples
python3 layer_norm.py

# List all available samples
python3 layer_norm.py --list
```

## Algorithm Principle

### 1. LayerNorm
The formula is as follows:
```
mean = mean(x, dim=-1)
var = var(x, dim=-1)
normalized = (x - mean) / sqrt(var + eps)
output = gamma * normalized + beta
```
It includes learnable scaling parameter `gamma` and shift parameter `beta`.

### 2. RMSNorm
The formula is as follows:
```
rms = sqrt(mean(x^2) + eps)
output = gamma * (x / rms)
```
It only includes the scaling parameter `gamma` and has lower computational cost.

## Key Technical Points

### Dynamic Shape Support
When handling variable Batch, you need to mark `dynamic_axis` and enable support in the JIT configuration:
```python
# Mark the Batch dimension as dynamic
x_pto = pypto.from_torch(x_torch, dynamic_axis=[0])

# Configure in the JIT kernel
def layer_norm_dynamic(...):
    ...
```

### Reduction Operations
Normalization layers extensively use the `pypto.sum` operator to compute the mean and sum of squares.

## Precautions
- **Precision**: For normalization operations, the numerical range of intermediate results (such as variance) may be large. Use FP32 precision for internal computation and convert back to FP16 or BF16 for the final output.
- **Tiling**: Set the tile shape based on the `hidden_size` to ensure that the vector units are fully utilized.
- **Verification**: All outputs in this sample are compared against the native PyTorch implementation for precision verification.
