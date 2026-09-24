# Attention Mechanism Sample

This sample demonstrates how to implement the scaled dot-product attention mechanism using PyPTO, which is the most core component of the Transformer architecture.

## Overview

This sample covers the entire process from basic attention computation to a complete multi-head attention implementation:
- **Scaled Dot-Product Attention**: Implements `softmax(Q @ K^T / sqrt(d_k)) @ V`.
- **Q, K, V Projection**: Shows how to project hidden states into queries, keys, and values.
- **Multi-Head Split and Concatenation**: Demonstrates tensor `reshape` and `transpose` operations to support parallel multi-head computation.
- **Dynamic Shape Support**: Supports variable batch size and sequence length.
- **Complete Attention Block**: A full implementation including input and output projections.

## Code Structure

- **`attention.py`**: Contains the core attention mechanism implementation logic, configuration class, and detailed test cases.

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
# Run all attention mechanism samples
python3 attention.py

# List all available samples
python3 attention.py --list
```

## Core Algorithm Implementation

### Scaled Dot-Product Attention

```python
@pypto.frontend.jit
def scaled_dot_product_attention(
    q: pypto.tensor((S1, DQK), pypto.DT_FP32),
    k: pypto.tensor((S2, DQK), pypto.DT_FP32),
    v: pypto.tensor((S2, DV), pypto.DT_FP32),
    output: pypto.tensor((S1, DV), pypto.DT_FP32)
):
    # 1. Compute Q @ K^T
    k_t = pypto.transpose(k, [0, 1, 3, 2])
    scores = pypto.matmul(q, k_t)

    # 2. Scale and Softmax
    scores_scaled = scores * scale
    attn_weights = pypto.softmax(scores_scaled, dim=-1)

    # 3. Apply to V
    output[:] = pypto.matmul(attn_weights, v)
```

## Key Technical Points

- **Efficient Transpose**: Uses `pypto.transpose` to achieve efficient multi-dimensional tensor rearrangement on the NPU.
- **Tiling Strategy**: Configures optimal `cube_tile_shapes` for large-scale matrix multiplications in the attention mechanism.
- **Dynamic Axis Marking**: Uses `dynamic_axis=[0, 2]` (Batch and SeqLen) to handle fluctuating input lengths during inference.

## Best Practices

- **Numerical Stability**: Scale before computing Softmax to prevent exponential explosion.
- **Memory Layout**: Keep K and V contiguous in memory as much as possible to optimize read speed.
- **Data Type**: Use BF16 precision when processing attention in large models.

## Precautions

- The memory complexity of the attention mechanism is O(N^2). Pay attention to memory usage for very long sequences.
- All implementations pass precision verification to ensure consistency with the PyTorch `scaled_dot_product_attention` result.
